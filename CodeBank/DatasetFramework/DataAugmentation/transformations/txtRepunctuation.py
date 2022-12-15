from typing import Union
import pandas as pd
from deepmultilingualpunctuation import PunctuationModel

class txtRepunctuation:
    """
    Sources:
    - [oliverguhr github](https://github.com/oliverguhr/deepmultilingualpunctuation)
    - [modelo hugging face](https://huggingface.co/kredor/punctuate-all)
    languages: 
      English, German, French, Spanish, Bulgarian, Italian, Polish, Dutch, Czech, Portugese, Slovak, Slovenian
    """
    def __init__(self):
        self.model = PunctuationModel(model = 'kredor/punctuate-all')
    def __call__(self, data:Union[str, pd.Series]):
        if isinstance(data, str): return self.model.restore_punctuation(txt)
        if isinstance(data, pd.Series): 
            raise NotImplementedError
            # for sample in data:
            #     return data.apply(self.model.restore_punctuation)


if __name__ == '__main__':
    test = 2
    original = """Cumpliendo con mi oficio piedra con piedra, pluma a pluma, pasa el invierno y deja sitios abandonados, habitaciones muertas: yo trabajo y trabajo, debo substituir tantos olvidos, llenar de pan las tinieblas, fundar otra vez la esperanza."""
    txt = """Cumpliendo con mi oficio piedra con piedra pluma a pluma pasa el invierno y deja sitios abandonados habitaciones muertas: yo trabajo y trabajo debo substituir tantos olvidos llenar de pan las tinieblas fundar otra vez la esperanza"""
    repunctuation = txtRepunctuation()
    if test == 1:
        print(f"test {test}: String repunctuation")
        result = repunctuation(txt)
        print(f"txt\n{txt}\n")
        print(f"result\n{result}\n")
        print(f"original\n{original}\n")
    if test == 2:
        print(f"test {test}: Dataframe repunctuation")
        data = pd.DataFrame({'src':txt})
        result = repunctuation(data['src'])
        print(data['src'])
        print(result)
        print(original)
        
