# coding: utf-8
# @File: path.py
# @Author: pixelock
# @Time: 2023/6/19 22:22

import os
from pathlib import Path

PATH_ROOT = Path(__file__).parent.parent

PATH_LOGS = PATH_ROOT.joinpath('logs')
PATH_LOG_SYSTEM = PATH_LOGS.joinpath('system.log')
PATH_LOG_SERVICE = PATH_LOGS.joinpath('service.log')
