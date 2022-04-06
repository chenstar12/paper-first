import logging
import os
import time

logger = logging.getLogger('')
file_name = os.path.join(os.getcwd(), 'log', time.strftime("%m%d-%H%M%S", time.localtime()) + '.txt')
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - [line:%(lineno)d] - : %(message)s')
file_handler = logging.FileHandler(file_name)
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
logger.info('info级别，一般用来打印一些正常的操作信息')
logger.warning('waring级别，一般用来打印警告信息')
logger.error('error级别，一般用来打印一些错误信息')
logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')
