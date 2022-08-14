import os, sys

class fileSystem:
    def __init__(self, path):
        self.path = path
        self.root = self.populate(path)

        self.currentDir = self.root

    # def __getitem__(self, key):
        # return self.fileSystem.in
    # def __iter__(self):
        # pass
    def __call__(self, command:str='.') -> dict:
        if command == '.': 
            dirs = [file[0][2:] for file in self.currentDir if isinstance(file, list)]
            files = [file for file in self.currentDir if not isinstance(file, list)]
            return {'dirs':dirs,'files':files}
        elif command == '~': self.currentDir = self.root

        elif command == 'files':
            return [file for file in self.currentDir if not isinstance(file, list)]
        elif command == 'dirs':
            return [file[0][2:] for file in self.currentDir if isinstance(file, list)]
        elif command.startswith('cd '):
            command = command[3:]
            for file in self.currentDir:
                if file[0][2:] == command:
                    self.currentDir = file[1:]
                    break
        else:
            raise NotImplementedError
            
    def __str__(self):
        def __str__(files:list, nTabs=0): #a recursive implementation
            txt=''
            for file in files:
                if isinstance(file, list):
                    fileName = file[0][2:]
                    txt += '\t'*nTabs + f"{fileName}\n"
                    txt += __str__(file[1:], nTabs+1)
                else:
                    txt += '\t'*nTabs + f"{file}\n"
            return txt
        return __str__(self.currentDir)

    def populate(self, path:str) -> list:
        """
        Note: 
            Excluding the root, the directories names are stored as the first item
        """
        files = []
        for file in os.listdir(path):
            if file.startswith('__'): continue
            if file.startswith('.'):  continue
            currentPath = path+'/'+file
            if os.path.isdir(currentPath): 
                files.append(['__'+file]+self.populate(currentPath))
            else:
                files.append(file)
        return files



if __name__ == '__main__':
    path = os.path.dirname(sys.modules['__main__'].__file__)
    path = os.path.split(path)[0] #cd ..
    f = fileSystem(path)
    print(f)
    print('\n')
    f('cd MachineLearning')
    print(f)
    #NotImplementedError
    # print('\n')
    # f('cd ..')
    # print(f)