# coding: utf-8
# @File: logger.py
# @Author: pixelock
# @Time: 2023/6/19 22:29

from loguru import logger

from configs.path import PATH_LOG_SYSTEM, PATH_LOG_SERVICE


def interface_filter(record):
    if 'requestId' in record['extra'] and record['extra']['requestId'] is not None:
        return True
    return False


logger.add(PATH_LOG_SYSTEM)
logger.add(
    PATH_LOG_SERVICE,
    format='[{time:YYYY-MM-DD HH:mm:ss.SSS}][{level}][{name}:{function}:{line}][{extra[requestId]}] {message}',
    filter=interface_filter,
)

__all__ = ['logger']
