#!/bin/bash

# this installs the virtualenv module
python3.7 -m pip install virtualenv
# this creates a virtual environment named "env"
python3.7 -m venv env
# this activates the created virtual environment
source env/bin/activate
# updates pip
pip3.7 install -U pip
# this installs the required python packages to the virtual environment
pip3.7 install -r requirements.txt

echo created environment
