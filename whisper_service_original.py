#!/usr/bin/env python3
#
# This script will submit itself to slurm and when it receives a USR1 it
# will submit itself again until all of the work has finished.
#
# The workdir is structured like this:
# /
#   .submit - the start of the sbatch command to submit ourselves
#   <work-to-do-directory>/  - client creates directory for each group of things to push
#     something
#     something
#
# It is important that the .submit file contains the start of a command that sets all 
# the values needed to submit the job properly and will trigger the batch script
# to receive a USR1 signal at least 30 seconds prior to termination.  
#
# The text of the .submit file is first run through string.Template, where references
# to environment variables and $workdir are resolved.  Then the command that's used
# for the service itself (basically this script plus options) is run using a
# shell.
#
# For slurm-based clusters, the command will look something like this:
#   sbatch --signal=B:USR1@30 -D $workdir -e $workdir/stderr.out -o $workdir/stdout.out
# (other options like GPU requirements, time limits, job names, etc can be added
# as necessary)

import activate_venv

import argparse
import atexit
import subprocess
from pathlib import Path
import logging
import sys
import yaml
import signal
import os
import time
from string import Template
import json

import whisper
import torch

# how long we should wait for the job files to settle before running them.
JOB_SETTLE_TIME=300

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Turn on debugging")
    parser.add_argument('workdir', help="Job Working Directory")
    parser.add_argument("--continuation", default=False, action='store_true', help="Continuation callback")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s [%(process)d:%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")

    workdir = Path(args.workdir)
    # do some sanity checking...
    if not workdir.is_dir():
        logging.error(f"The working directory {workdir} isn't a directory")
        exit(1)

    # there needs to be a .submit file in the working directory that tells us
    # how this working directory resubmits itself.
    if not (workdir / ".submit").exists():
        logging.error(f"The working direcory doesn't contain a .submit file")
        if not (workdir / ".submit.sample").exists():
            logging.info("Creating a sample .submit file")
            with open(workdir / ".submit.sample", "w") as f:
                f.write("sbatch --signal B:USR1@30 -D$workdir \\\n")
                f.write("   -e $workdir/stderr.out -o $workdir/stdout.out \\\n")
                f.write("   --open-mode=append --parsable \\\n")
                f.write("   --mail-type=ALL \\\n")
                f.write("   -p general\n")
        exit(1)

    if not args.continuation:
        # This isn't a continuation.  Check to see if we're already submitted 
        # or actually running...  
        # There's a race condition here - the period between checking for 
        # jobinfo.yaml and creating the file in self_submission needs to be
        # protected.  We'll do this with an exclusive open.  This isn't an
        # issue with end-of-task resubmission because of the jobinfo.yaml gate
        try:
            with open(workdir / "submit.lock", "x") as lockfile:
                atexit.register(lambda: (workdir / "submit.lock").unlink(missing_ok=True))
                if not (workdir / "jobinfo.yaml").exists():
                    # the jobinfo doesn't exist so this is a submission event
                    info = vars(args)
                    info['jobid'] = None
                    self_submission(workdir, info)
                    logging.info(f"Job initially submitted with jobid {info['jobid']}")
                else:
                    logging.info("Jobinfo file already exists, we're already in the system")        
            (workdir / "submit.lock").unlink(missing_ok=True)
        except FileExistsError:
            logging.info("Submission conflict")
        exit(0)
    else:
        # load our job information
        with open(workdir / "jobinfo.yaml") as f:
            info = yaml.safe_load(f)

    # register the USR1 signal handler so we can shut down gracefully and
    # resubmit ourselves if needed.
    def usr1handler(sig, frame):
        """Handle USR1"""
        # resubmit ourselves.
        self_submission(workdir, info)
        exit(0)

    signal.signal(signal.SIGUSR1, usr1handler)

    # DO THE WORK!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Whisper will use {device} for computation")
    while True:
        todo = get_todo_list(workdir)
        if not todo:
            break
        # if none of the items in todo are "ready" then we're going to sleep a bit
        logging.info(f"Starting processing of {len(todo)} jobs")

        # filter out everything that isn't settled...
        todo = [x for x in todo if x['settled']]
        if not todo:
            logging.info("Waiting for jobs to settle.")
            time.sleep(JOB_SETTLE_TIME)
            continue

        # get the models we need to use...
        process_count = 0
        models = {x['model'] for x in todo}
        for model_name in models:
            logging.info(f"Starting processing for jobs using {model_name} model")
            model = whisper.load_model(model_name, device,)
            for t in todo:
                if t['model'] != model_name:
                    continue
                logging.info(f"Processing job in {t['jobdir']}")
                for f in t['manifest']:
                    try:
                        logging.info(f"Starting processing of {f}")
                        audio = whisper.load_audio(t['jobdir'] / f)

                        if 'prompt' not in t:
                            t['prompt'] = None

                        if 'language' not in t or t['language'] == 'auto':
                            probable_languages = ('en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja')                
                            detect_audio = whisper.pad_or_trim(audio)
                            # This is a bug https://github.com/openai/whisper/discussions/1778
                            # basically, the v3 large model uses 128 mels, and the others use 80.
                            mel = whisper.log_mel_spectrogram(detect_audio, n_mels=model.dims.n_mels).to(device)
                            _, probs = model.detect_language(mel)            
                            probs = {k: v for k, v in probs.items() if k in probable_languages}                        
                            language = max(probs, key=probs.get)                          
                            logging.info(f"Language detection: {language}:  {probs}")
                        else:
                            language = t['language']

                        start = time.time()                    
                        res = whisper.transcribe(model, audio, word_timestamps=True, language=language,
                                                verbose=None, initial_prompt=t['prompt'])
                        duration = time.time() - start

                        res['_job'] = {
                            'runtime': duration,
                            'device': device,
                            'language': language,
                            'model': t['model'],
                            'prompt': t['prompt']
                        }

                        with open(t['jobdir'] / (f + ".whisper.json"), "w") as f:
                            json.dump(res, f, indent=4)
                        process_count += 1

                    except Exception as e:
                        logging.exception(f"Failed during transcription of {f}")

            # unload the model to the best of our ability
            del(model)
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        if process_count == 0:
            logging.warning("Some processes didn't complete even though they were valid")
            break

    logging.debug("Finished processing things.  Shutting down.")
    # Remove the jobinfo file to that future calls will restart the whole game
    (workdir / "jobinfo.yaml").unlink(missing_ok=True)


job_warnings = set()
def get_todo_list(workdir: Path) -> list:
    results = []
    for jobfile in workdir.glob("*/whisper.job"):
        try:
            with open(jobfile) as f:
                jobdata = yaml.safe_load(f)
        except Exception as e:
            if jobfile not in job_warnings:
                job_warnings.add(jobfile)
                logging.warning(f"Cannot read {jobfile}: {e}")
            continue

        # update the core data 
        jobdata.update({'jobdir': jobfile.parent,
                        'settled': (time.time() - jobfile.stat().st_mtime) > JOB_SETTLE_TIME})
        
        # update the manifest.  If there's a .whisper.json file for a manifest
        # entry, we'll remove the entry.  If there's nothing left in the manifest
        # then we won't add it to the jobs since it's effectively finished.
        for file in list(jobdata['manifest']):
            if (jobdata['jobdir'] / (file + ".whisper.json")).exists():
                jobdata['manifest'].remove(file)
        if not jobdata['manifest']:
            logging.debug(f"The job file {jobfile.resolve()} is finished, skipping")
            continue                
        
        
        results.append(jobdata)
    return results


def self_submission(workdir: Path, info: dict):
    """Submit ourselves back into slurm"""
    # read the .submit file for the command to use for submission
    submit_command = (workdir / ".submit").read_text().strip()
    template_vars = {'workdir': str(workdir.resolve())}
    template_vars.update(os.environ)
    template = Template(submit_command)
    submit_command = template.safe_substitute(template_vars)
    submit_array = [submit_command,
                    str(Path(__file__).resolve()),
                    str(workdir.resolve()),
                    '--continuation']
    if info['debug']:
        submit_array.append("--debug")

    logging.debug(f"Submitting job with command {' '.join(submit_array)}")
    p = subprocess.run(" ".join(submit_array), 
                       stdout=subprocess.PIPE, encoding="utf-8",
                       shell=True)
    if p.returncode != 0:
        logging.error(f"Couldn't submit ourselves, rc {p.returncode}: {p.stdout}")
        exit(1)
    
    info['jobid'] = p.stdout.strip()
    with open(workdir / "jobinfo.yaml", "w") as f:
        yaml.safe_dump(info, f)


if __name__ == "__main__":
    main()