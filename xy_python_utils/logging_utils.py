#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 10, 2015.

import logging
import sys

class DebugOrInfoFilter(logging.Filter):
    """Keep the record only if the level is debug or info."""
    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.INFO)

def split_stdout_stderr(logger, formatter):
    """Configure the logger such that debug and info messages are directed to stdout,
    while more critical warnings and errors to stderr.
    """
    stdoutHandler = logging.StreamHandler(sys.stdout)
    stdoutHandler.setLevel(logging.DEBUG)
    stdoutHandler.setFormatter(formatter)
    stdoutHandler.addFilter(DebugOrInfoFilter())
    logger.addHandler(stdoutHandler)

    stderrHandler = logging.StreamHandler(sys.stderr)
    stderrHandler.setLevel(logging.WARNING)
    stderrHandler.setFormatter(formatter)
    logger.addHandler(stderrHandler)
