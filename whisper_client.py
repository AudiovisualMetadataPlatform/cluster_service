#!/usr/bin/env python3

import activate_venv
import argparse
import paramiko
import logging
import getpass
import uuid
from pathlib import Path
import yaml
from stat import S_ISDIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Turn on debugging")
    parser.add_argument("--hpchost", type=str, default="localhost", help="HPC host")
    parser.add_argument("--hpcuser", type=str, default=getpass.getuser(), help="HPC User")
    parser.add_argument("--hpcworkdir", type=str, required=True, help="HPC Working directory")
    parser.add_argument("--hpcscript", type=str, default="cluster_service/whisper_service.py")
    sps = parser.add_subparsers(required=True, metavar='mode', dest='mode', help="Client Mode")
    
    sp = sps.add_parser('submit', help="Submit a new job")
    sp.add_argument("--model", choices=['base', 'tiny', 'small', 'medium', 'large', 'large_v2', 'large_v3'], default='medium', help="Language model")
    sp.add_argument("--language", choices=['auto', 'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja'], default="en", help="Language")
    sp.add_argument("--prompt", default=None, help="Model prompt")
    sp.add_argument("files", nargs='+', help="Files to process")
    
    sp = sps.add_parser("list", help="List processing jobs")

    sp = sps.add_parser('check', help="Check the status of a job")
    sp.add_argument("jobid", help="Job ID")

    sp = sps.add_parser('retrieve', help="Retrieve the results of a job")
    sp.add_argument("jobid", help="Job ID")
    sp.add_argument("outdir", help="Ouput directory")
    sp.add_argument("--purge", default=False, action="store_true", help="Purge after retrieval")

    sp = sps.add_parser('purge', help="Remove a job from the server")
    sp.add_argument("jobid", help="Job ID")

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if args['debug'] else logging.INFO,
                    format="%(asctime)s [%(process)d:%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")
    # tell paramiko that I don't want messages unless they're important, even if
    # I've set the root logger to debug.
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    modes = {'submit': submit_job,
             'list': list_jobs,
             'check': check_job,
             'retrieve': retrieve_job,
             'purge': purge_job}
    try:
        # Everyone needs an ssh connection, so let's make it here.
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(args['hpchost'], username=args['hpcuser'])

        # call the right mode
        modes[args['mode']](ssh, **args)
        exit(0)
    except Exception as e:
        logging.exception(f"Failed in mode {args['mode']}: {e}")
        exit(1)


def submit_job(ssh: paramiko.SSHClient, hpcworkdir=None, model=None, 
               language=None, prompt=None, files=[], hpcscript=None, **kwargs):
    """Create a new job on the HPC cluster"""        

    sftp = ssh.open_sftp()
    sftp.chdir(hpcworkdir)    
    
    # create the job directory in the hpc workspace...
    jobname = str(uuid.uuid4())
    sftp.mkdir(jobname)
    sftp.chdir(jobname)
    logging.info(f"Created job directory {jobname}")

    # copy the individual files to the job directory
    config = {
        'manifest': [],
        'language': language,
        'prompt': prompt,
        'model': model
    }
    for f in files:
        f = Path(f)
        logging.info(f"Uploading {f}")
        sftp.put(str(f.resolve()), f.name)
        config['manifest'].append(f.name)

    logging.info("Creating job file")
    # write the whisper.job parameters file.
    file = sftp.file('whisper.job', 'w', -1)
    yaml.safe_dump(config, file)

    # tell the system we've got something new to do
    logging.info("Submitting the job to HPC")
    debug = "--debug" if kwargs['debug'] else ''
    command = f'bash -c "{hpcscript} {debug} {hpcworkdir}; echo \\$?"'
    logging.debug(f"Submit command: {command}")
    (_, stdout, stderr) = ssh.exec_command(command)
    # the last line of stdout should be the return code from the command.
    sout = [x.strip() for x in stdout.readlines()]
    if not sout or sout[-1] != '0':
        logging.error("Submission to HPC failed:\n" + "".join(stderr.readlines())) 
        purge_job(ssh, hpcworkdir=hpcworkdir, jobid=jobname)
        exit(1)
    else:
        logging.debug("Submission stdout:\n" + '\n'.join(sout) + "\nstderr:\n" + "".join(stderr.readlines()))

    # tell the caller we're good
    print(jobname)
    exit(0)


def list_jobs(ssh, hpcworkdir=None, **kwargs):
    "Print the jobids on the system"
    # connect to the hpc node.
    sftp = ssh.open_sftp()
    for f in sftp.listdir(hpcworkdir):
        if valid_job(sftp, hpcworkdir, f):
            print(f)
        
    exit(0)


def check_job(ssh: paramiko.SSHClient, hpcworkdir=None, jobid=None, **kwargs):
    "Return the status of the job"
    sftp = ssh.open_sftp()
    if not valid_job(sftp, hpcworkdir, jobid):
        logging.error(f"Cannot check job: Jobid {jobid} is not valid")
        exit(1)

    # read the whisper.job file to get the manifest
    job = yaml.safe_load(sftp.open(f"{hpcworkdir}/{jobid}/whisper.job").read())
    for n in job['manifest']:
        try:
            sftp.stat(f"{hpcworkdir}/{jobid}/{n}.whisper.json")
        except:
            print("IN PROGRESS")
            exit(2)
    print("FINISHED")
    exit(0)
    
    
    





def retrieve_job(ssh: paramiko.SSHClient, hpcworkdir=None, jobid=None, outdir=None, purge=None, **kwargs):
    "Retrive the directory from host"    
    sftp = ssh.open_sftp()    
    if not valid_job(sftp, hpcworkdir, jobid):
        logging.error(f"Cannot retrieve job: Jobid {jobid} is not valid")
        exit(1)

    file_list = recursive_list(sftp, f"{hpcworkdir}/{jobid}")
    for f in [x for x in file_list if x.endswith('.whisper.json')]:
        fname = f.split('/')[-1]
        logging.info(f"Retrieving {f}")
        sftp.get(f, outdir + "/" + fname)        

    if purge:
        # remove the job and directory
        purge_job(ssh, hpcworkdir, jobid)

    exit(0)


def purge_job(ssh: paramiko.SSHClient, hpcworkdir=None, jobid=None, **kwargs):
    sftp = ssh.open_sftp()
    if not valid_job(sftp, hpcworkdir, jobid):
        logging.error(f"Cannot purge job: Jobid {jobid} is not valid")
    jobdir = f"{hpcworkdir}/{jobid}"
    logging.warning(f"Purging job directory at {jobdir}")
    file_list = recursive_list(sftp, jobdir)
    for f in file_list:
        if f.endswith("/"):
            logging.debug(f"RMDIR: {f}")
            sftp.rmdir(f)
        else:
            logging.debug(f"UNLINK: {f}")
            sftp.unlink(f)
    # remove the job directory
    logging.debug(f"UNLINK {jobdir}")
    sftp.rmdir(jobdir)



def valid_job(sftp: paramiko.SFTPClient, workdir, jobid):
    "return true or false if the jobid is valid"
    try:
        # make sure the job directory exists
        jobdir = f"{workdir}/{jobid}"
        s = sftp.stat(jobdir)
        if not S_ISDIR(s.st_mode):
            logging.debug(f"{jobdir} isn't a directory")
            return False
        # make sure we have a whisper.job file
        s = sftp.stat(f"{jobdir}/whisper.job")
        return True
    except Exception as e:
        logging.debug(f"Couldn't check for valid job: {e}")
        return False
    


def recursive_list(sftp, path):
    """Return a list of all of the (file) paths rooted at the given path"""
    results = []
    logging.debug(f"Scanning {path}")    
    for item in sftp.listdir_attr(path):        
        if S_ISDIR(item.st_mode):
            results.extend(recursive_list(sftp, f"{path}/{item.filename}"))            
            results.append(f"{path}/")
        else:                
            results.append(f"{path}/{item.filename}")
    return results


if __name__ == "__main__":
    main()