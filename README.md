## Creating the virtual Environment
The virtual environment is created with the `create_venv.sh` script.  This script will
create a python venv and install all of the requirements for the software

The software was written with python 3.11 and it should be used if possible.
The `create_venv.sh` script will search for python 3.12 -> 3.9 and used the
first version found.

### Installing on BigRed 200 at IU
In order for the virtual environment to be created properly it is necessary to
load python using: `module load python/3.11` before running `create_venv.sh`

After the venv has been created, the module doesn't need to be loaded as the
python files copied into .venv will be used.

