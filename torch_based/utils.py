# @Time    : 11/16/2019 11:53 AM
# @Author  : mikelkl
import time
import logging

from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))
