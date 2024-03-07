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
python3.$python_version -m venv --copies $SCRIPT_DIR/.venv

# install packages
source $SCRIPT_DIR/.venv/bin/activate
pip3.$python_version install --upgrade pip
pip3.$python_version install -r $SCRIPT_DIR/requirements.txt
