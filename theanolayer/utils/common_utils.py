# author: Hao-ran Wei
# e-mail: whr94621@gmail.com

import time
import sys

__all__ = [
    'PRINT',
    'INFO',
    'WARNING',
    'ERROR',
    'scope'
]

def PRINT(*string):
    ss = [s if isinstance(s, str) else '{0}'.format(s) for s in string]
    sys.stderr.write('{0}\n'.format(' '.join(ss)))

def INFO(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    sys.stderr.write('{0} INFO:{1}\n'.format(time.strftime(time_format), string))

def WARNING(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    sys.stderr.write('{0} WARNING:{1}\n'.format(time.strftime(time_format), string))

def ERROR(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    sys.stderr.write('{0} ERROR:{1}\n'.format(time.strftime(time_format), string))

def scope(outer, name):
    return '{0}/{1}'.format(outer, name)