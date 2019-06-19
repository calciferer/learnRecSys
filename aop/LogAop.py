# 记录方法调用信息、执行时间等
import functools
import logging
import string
import time


def log():
    def log(func):
        @functools.wraps(func)
        def wrapper(*args, **kv):
            logs = []
            logs.append("-------\n")
            logs.append("文件:%s\n" % func.__globals__['__file__'])
            logs.append("方法名：%s\n" % func.__name__)
            logs.append("参数:args:%s\n参数kv:%s\n" % (args, kv))
            startTime = time.time()
            f = func(*args, **kv)
            logs.append("返回值:%s\n" % str(f))
            logs.append("用时：%.2fms\n" % ((time.time() - startTime) * 1000))
            logs.append("--------")
            logging.debug("".join(logs))
            return f

        return wrapper

    return log
