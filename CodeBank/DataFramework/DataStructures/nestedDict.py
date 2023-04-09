
"""
TODO:
  diccionario que accede al estilo archivo
  c/a/b/c
  c_a_b_c
  para acceder a diccionarios anidados desde el key

TODO:
    implementar el 
    with print:
        2+2
        a**2

    todo lo que salga, va por stdout
"""
import regex

class NestedDict(dict):
    """NestedDict is a dictionary of NestedDict.
    The idea is to traverse the data through a regular expression
    Example 1:
        nd = NestedDict()
        nd['/home'] = {'usr':'myName'}
        nd['/home] = {'local': 'media'}
        
        with print:
            nd['/home/usr'] #myName
            nd['/home/local] #media
    """
    def __init__(self, *args, delimiter='_',**kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter

    def __setitem__(self, key:str, value:str):
        print('set')
        super().__setitem__(key, value)

    def __getitem__(self, key:str):
        print('get')
        return super().__getitem__(key)




if __name__ == '__main__':
    struct = NestedDict()

    struct['hola'] = 'f'
    print(struct['hola'])

    