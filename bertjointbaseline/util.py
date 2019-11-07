#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 11/8/2019 8:52 AM
# @Author  : mikelkl
from datetime import datetime, timedelta, timezone

def get_BJ_time():
    # 拿到UTC时间，并强制设置时区为UTC+0:00
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    # astimezone()将转换时区为北京时间
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    current_time = bj_dt.strftime('%m%d_%H-%M-%S')

    return current_time