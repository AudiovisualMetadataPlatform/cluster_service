# This is a bootstrap to start up the venv from any python.  The gist of it
# is that any python3 will be able to run this code and the paths will will
# be updated and then the script re-exec'd which will pick up the python
# and venv that we really want, rather than what was present elsewhere.
import __main__
import os
from pathlib import Path
import platform
import sys
import time

debug = 'ACTIVATE_DEBUG' in os.environ

def update_env(key: str, val: str, delim: str = ":"):
    val = str(val)
    if key in os.environ:
        os.environ[key] = val + delim + os.environ[key]
    else:
        os.environ[key] = val

# build ld library path
def update_lib_path():
    paths = set()
    for pat in ("*.so", "*.so.[0-9]*"):
        for p in Path(sys.path[0], ".venv").glob("**/" + pat):
            paths.add(p.parent)
    for p in paths:
        update_env('LD_LIBRARY_PATH', p)
    if debug:
        print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

if debug:
    print("Using Python Version:", platform.python_version_tuple())

if 'VENV_RESTART' not in os.environ:
    # Since the VENV_RESTART sentinel isn't present it means we haven't run
    # this yet.  Let's set up the environment.
    venv_path = Path(sys.path[0], '.venv').resolve()
    # set the VIRTUAL_ENV
    update_env("VIRTUAL_ENV", venv_path)
    # update the PATH
    update_env("PATH", venv_path / 'bin')
    # sentinel to know we've already been here.
    update_env("VENV_RESTART", 1)
    # update the ld_library_path
    update_lib_path()
    # put the script path into the PYTHONPATH
    update_env("PYTHONPATH", sys.path[0])

    # update any environment variables we may have...
    for p in (venv_path / "environ").glob('*'):
        update_env(p.name, p.read_text().strip())


    # build the command line...
    this_script = str(Path(__main__.__file__).resolve())
    #args = [this_script, *sys.argv]
    if debug:
        print("Preparing to re-exec", os.getpid())
        print(f"args: {this_script} {sys.argv}")    
        time.sleep(1)
    os.execve(this_script, sys.argv, os.environ)
    
else:
    if debug:
        print("Second go-round", os.getpid())
        print("PATH:", os.environ['PATH'])
        print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])