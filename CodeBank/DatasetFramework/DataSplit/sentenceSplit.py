import pandas as pd

class sentenceSplit:
    """
    Given a dataframe of text it splits into a list of sentences strings
    """
    def __init__(self):
        self.delimiter  = '. '

    def __call__(self, data:pd.DataFrame) -> pd.DataFrame:
        print(data)





if __name__ == '__main__':
    test = 1
    data = {''}
    data = pd.DataFrame(data)
    if test == 1:
        print(f"test {test}: split a text into a list of strings")