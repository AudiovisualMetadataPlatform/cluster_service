#!/bin/env python3
import platform
print("Current version", platform.python_version_tuple())

import activate_venv
import yaml
from pathlib import Path
import os
import sys
import time
import torch

print("Torch OK!")