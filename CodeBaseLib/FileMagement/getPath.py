"""
Used by
    - datasetNDD/experimentNDD
"""

import os, sys


class getPath:
    """
    Returns the path of the main file
    """
    def __init__(self,relativePath:str='', moduleName='__main__'):
        self.relativePath = relativePath
        self.moduleName = moduleName
        self.globalPath = os.path.dirname(sys.modules[self.moduleName].__file__)+'/'
        
    def calcMode(self, permission:dict) -> dict:
        mode = permission['exe']+permission['write']*2+permission['read']*4
        mode = (permission['owner']+permission['group']*8+permission['other']*64)*mode
        return mode

    def __call__(self, relativePath:str, permission={'exe':True, 'read':True, 'write':True, 'owner':True, 'group':True, 'other':True}):
        path = self.globalPath + self.relativePath + relativePath

        if path[-1] == '/': dir, file = path, None
        else :              dir, file = os.path.split(path) #split in dir and path
        self.dirExists(dir, permission)
        return path

    def dirExists(self, path, permission):
        if not os.path.exists(path): #already exists?
            mode = self.calcMode(permission)
            os.mkdir(path, mode)


if __name__ == '__main__':
    test = 2
    if test == 1:
        permission = {'exe':True, 'read':True, 'write':True, 'owner':True, 'group':True, 'other':True}
        mode = permission['exe']+permission['write']*2+permission['read']*4
        mode = (permission['owner']+permission['group']*8+permission['other']*64)*mode
        print(oct(mode))
    if test == 2:
        path = getPath('test/')
        dir = path('')
        print(f"your file is:\n{path('__init__.py')}\nExist = {os.path.exists(dir)}")
        os.rmdir(dir)
        print(f"removed: \n{dir}\nExist = {os.path.exists(dir)}")
