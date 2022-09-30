import logging

import time
import regex

import os
from CodeBaseLib.FileMagement import getPath

class logger:
    """
    logs a message to the output or a file

    <format>:str = '%Y-%m-%d %H:%M:%S %(message)'

        date options:
            %Y-%m-%d-%H-%M-%S (year, month, day, Hours, Minutes, Seconds)

    <filename>:str = None   
        path to a file, by default it prints in the terminal

    """

    def __init__(self, format:str='%Y-%m-%d %H:%M:%S|| %(message)',filename:str=None):
        self.format = format
        self.filename = filename
        self.path = getPath()

        self.writer = print if filename==None else self.fileWriter

    def fileWriter(self, message):
        path = self.path(self.filename)
        with open(path, 'a') as f:
            f.write(message+'\n')

    def __call__(self, message:str):
        format = time.strftime(self.format)

        format = format.replace('%(message)', message)

        self.writer(format)


if __name__=='__main__':
    test = 1
    if test == 1:
        log= logger()
        log('hello world')
        log('Fine')
        # logging.basicConfig()
        # logging.log(1,msg='hola')
    if test == 2:
        log= logger(filename='hola.txt')
        for i in range(1):
            log(f'{i}: hello world')
            time.sleep(1)

        p = getPath()
        path = p('hola.txt')
        os.remove(path)