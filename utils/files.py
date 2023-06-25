# coding: utf-8
# @File: files.py
# @Author: pixelock
# @Time: 2023/6/25 23:25

import hashlib

from logger import logger


def checksum(file_path, file_hash):
    with open(file_path, "rb") as datafile:
        binary_data = datafile.read()
    sha1 = hashlib.sha1(binary_data).hexdigest()
    if sha1 != file_hash:
        logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))
