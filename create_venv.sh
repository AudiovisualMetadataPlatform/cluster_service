#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

found_python=0
for python_version in 12 11 10 9; do 
    if which python3.$python_version >/dev/null 2>&1; then
        found_python=1
        break
    fi
done

if [ $found_python == 0 ]; then
    echo Cannot find a valid version of python3
    exit 1
fi

# create the VENV
echo Using python3.$python_version
python3.$python_version -m venv $SCRIPT_DIR/.venv

# install packages
source $SCRIPT_DIR/.venv/bin/activate
pip3.$python_version install --upgrade pip
pip3.$python_version install -r $SCRIPT_DIR/requirements.txt

# save the values of PATH and LD_LIBRARY_PATH so they'll be available when we 
# use the activate_venv.py module.
mkdir $SCRIPT_DIR/.venv/environ
for n in PATH LD_LIBRARY_PATH; do 
    echo ${$n} > $SCRIPT_DIR/.venv/environ/$n
done