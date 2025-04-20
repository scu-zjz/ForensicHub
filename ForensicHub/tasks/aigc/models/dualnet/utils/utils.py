import logging
import os

class Logger:
    def __init__(self, name, path, Clevel = logging.DEBUG, Flevel=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s','%Y-%m-%d %H:%M:%S')

        sh = logging.StreamHandler()
        sh.setLevel(Clevel)
        sh.setFormatter(formatter)

        fh = logging.FileHandler(path)
        fh.setLevel(Flevel)
        fh.setFormatter(formatter)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)

    def war(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def cri(self,message):
        self.logger.critical(message)
