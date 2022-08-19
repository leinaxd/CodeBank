"""
Used by
    - datasetNDD/experimentNDD
"""

import os, sys


def getPath(moduleName='__main__')->str:
    """
    Returns the path of the main file
    """
    return os.path.dirname(sys.modules[moduleName].__file__)+'/'