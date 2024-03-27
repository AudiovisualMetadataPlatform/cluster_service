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
import yaml
import signal
import os
import time
from string import Template
import __main__

class ClusterService():
    def __init__(self, workdir: Path = None, settle_time=300):
        self.JOB_SETTLE_TIME=settle_time
        self.workdir = workdir


    def work(self):
        raise NotImplementedError("You must implement the work method")


    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true", default=False, help="Turn on debugging")
        parser.add_argument('workdir', help="Job Working Directory")
        parser.add_argument("--continuation", default=False, action='store_true', help="Continuation callback")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                            format="%(asctime)s [%(process)d:%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")
        # turn off numba debug messags
        logging.getLogger('numba').setLevel(logging.WARNING)    



        self.workdir = Path(args.workdir)
        # do some sanity checking...
        if not self.workdir.is_dir():
            logging.error(f"The working directory {self.workdir} isn't a directory")
            exit(1)

        # there needs to be a .submit file in the working directory that tells us
        # how this working directory resubmits itself.
        if not (self.workdir / ".submit").exists():
            logging.error(f"The working direcory doesn't contain a .submit file")
            if not (self.workdir / ".submit.sample").exists():
                logging.info("Creating a sample .submit file")
                with open(self.workdir / ".submit.sample", "w") as f:
                    f.write("sbatch --signal B:USR1@30 -D$workdir \\\n")
                    f.write("   -e $workdir/stderr.out -o $workdir/stdout.out \\\n")
                    f.write("   --open-mode=append --parsable \\\n")
                    f.write("   --mail-type=ALL \\\n")
                    f.write("   -p general\n")
            exit(1)

        if args.continuation and not (self.workdir / "jobinfo.yaml").exists():
            # somehow we got called with the continuation flag but we're not
            # really running.  Probably an accident, so let's ignore the
            # continuation flag
            args.continuation = False

        if not args.continuation:
            # This isn't a continuation.  Check to see if we're already submitted 
            # or actually running...  
            # There's a race condition here - the period between checking for 
            # jobinfo.yaml and creating the file in self_submission needs to be
            # protected.  We'll do this with an exclusive open.  This isn't an
            # issue with end-of-task resubmission because of the jobinfo.yaml gate
            try:
                with open(self.workdir / "submit.lock", "x") as lockfile:
                    atexit.register(lambda: (self.workdir / "submit.lock").unlink(missing_ok=True))
                    if not (self.workdir / "jobinfo.yaml").exists():
                        # the jobinfo doesn't exist so this is a submission event
                        info = vars(args)
                        info['jobid'] = None
                        self.self_submission(info)
                        logging.info(f"Job initially submitted with jobid {info['jobid']}")
                    else:
                        logging.info("Jobinfo file already exists, service is already in the system")        
                (self.workdir / "submit.lock").unlink(missing_ok=True)
            except FileExistsError:
                logging.info("Submission conflict")
            exit(0)
        else:
            # load our job information
            with open(self.workdir / "jobinfo.yaml") as f:
                info = yaml.safe_load(f)

        # register the USR1 signal handler so we can shut down gracefully and
        # resubmit ourselves if needed.
        def usr1handler(sig, frame):
            """Handle USR1"""
            # resubmit ourselves.
            self.self_submission(info)
            exit(0)

        signal.signal(signal.SIGUSR1, usr1handler)

        # DO THE WORK!
        self.work()

        logging.info("Finished processing things.  Shutting down.")
        # Remove the jobinfo file to that future calls will restart the whole game
        (self.workdir / "jobinfo.yaml").unlink(missing_ok=True)


    def filter_jobdata(self, jobdata: dict):
        """Remove files from the manifest that are already finished"""
        pass


    def get_todo_list(self, jobfile:str) -> list:
        results = []
        for jobfile in self.workdir.glob(f"*/{jobfile}"):
            try:
                with open(jobfile) as f:
                    jobdata = yaml.safe_load(f)
            except Exception as e:
                if jobfile not in self.job_warnings:
                    self.job_warnings.add(jobfile)
                    logging.warning(f"Cannot read {jobfile}: {e}")
                continue

            # update the core data 
            jobdata.update({'jobdir': jobfile.parent,
                            'settled': (time.time() - jobfile.stat().st_mtime) > self.JOB_SETTLE_TIME})
            
            # filter the jobfile to see if there's anything todo in the manifest.
            self.filter_jobdata(jobdata)

            if not jobdata['manifest']:
                logging.debug(f"The job file {jobfile.resolve()} is finished, skipping")
                continue                
                    
            results.append(jobdata)

        # filter out everything that isn't settled...
        while True:
            todo = [x for x in results if x['settled']]
            if results and not todo:
                # we have results but none of them are ready.  Wait a bit and
                # try again.
                logging.info("Waiting for jobs to settle.")
                time.sleep(self.JOB_SETTLE_TIME)
                continue
            # so either we don't have results or we have something todo...so
            # drop out.
            break

        return results    


    def self_submission(self, info: dict):
        """Submit ourselves back into slurm"""
        # read the .submit file for the command to use for submission
        submit_command = (self.workdir / ".submit").read_text().strip()
        template_vars = {'workdir': str(self.workdir.absolute())}
        template_vars.update(os.environ)
        template = Template(submit_command)
        submit_command = template.safe_substitute(template_vars)
        
        command_array = [str(Path(__main__.__file__).absolute()),
                        str(self.workdir.absolute()),
                        '--continuation']
        if info['debug']:
            command_array.append("--debug")
        slurm_script = '\n'.join(['#!/bin/bash',
                                  ' '.join(command_array),
                                  'exit $?'])
        
        logging.debug(f"Submission command:\n{submit_command}")
        logging.debug(f"Submitting job with slurm script:\n{slurm_script}")
        p = subprocess.run(submit_command, input=slurm_script,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                           shell=True)
        if p.returncode != 0:
            logging.error(f"Couldn't submit ourselves, rc {p.returncode}: {p.stderr}")
            exit(1)

        info['jobid'] = p.stdout.strip()
        with open(self.workdir / "jobinfo.yaml", "w") as f:
            yaml.safe_dump(info, f)


