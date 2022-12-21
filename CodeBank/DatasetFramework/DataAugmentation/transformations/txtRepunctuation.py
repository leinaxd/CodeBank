from typing import Union
import pandas as pd
try:
    from deepmultilingualpunctuation import PunctuationModel
except:
    print('deepmultilingualpunctuation not installed. ignoring')

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
        if isinstance(data, str): return self.model.restore_punctuation(data)
        if isinstance(data, (pd.Series,list)):
            out = []
            for sample in data:
                # return data.apply(self.model.restore_punctuation)
                out.append(self.model.restore_punctuation(sample))
            if isinstance(data, list): return out
            else:                      return pd.Series(out)


if __name__ == '__main__':
    test = 4
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
        data = pd.DataFrame({'src':[txt]})
        for sample in  data['src']:
            result = repunctuation(sample)
            print(f"sample\n{sample}\n")
            print(f"result\n{result}\n")
            print(f"original\n{original}\n")
    if test == 3:
        print(f"test {test}: Dataframe repunctuation")
        data = pd.DataFrame({'src':[txt]})
        result = repunctuation(data['src'])
        print(f"sample\n{data['src']}\n")
        print(f"result\n{result}\n")
        print(f"original\n{original}\n")
    if test == 4:
        print(f"test {test}: list repunctuation")
        data = [txt]
        result = repunctuation(data)
        print(f"sample\n{data}\n")
        print(f"result\n{result}\n")
        print(f"original\n{original}\n")
        
