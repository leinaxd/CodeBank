
import torch
import pandas as pd


from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRandomSampling
from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRepunctuation
from CodeBank.DatasetFramework.DataAugmentation.transformations import txtSynonyms
from BackTranslation import BackTranslation

import time

class txtAugmentation:
    """
    !pip install BackTranslation
    sources:
        https://github.com/hhhwwwuuu/BackTranslation
        https://neptune.ai/blog/data-augmentation-nlp
    transformations: Dict[type:str, prob:float]
        - synonyms
        - BackTranslation: language
            chinese, english, spanish
            'af': 'afrikaans',               'sq': 'albanian', 
            'am': 'amharic',                 'ar': 'arabic',                    'hy': 'armenian', 
            'az': 'azerbaijani',             'eu': 'basque',                    'be': 'belarusian',              
            'bn': 'bengali',                 'bs': 'bosnian',                   'bg': 'bulgarian', 
            'ca': 'catalan',                 'ceb': 'cebuano',                  'ny': 'chichewa',
            'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)',  'co': 'corsican', 
            'hr': 'croatian',                'cs': 'czech',                     'da': 'danish',
            'nl': 'dutch',                   'en': 'english',                   'eo': 'esperanto', 
            'et': 'estonian',                'tl': 'filipino',                  'fi': 'finnish', 
            'fr': 'french',                  'fy': 'frisian',                   'gl': 'galician',                
            'ka': 'georgian',                'de': 'german',                    'el': 'greek', 
            'gu': 'gujarati',                'ht': 'haitian creole',            'ha': 'hausa', 
            'haw': 'hawaiian',               'iw': 'hebrew',                    'he': 'hebrew', 
            'hi': 'hindi',                   'hmn': 'hmong',                    'hu': 'hungarian', 
            'is': 'icelandic',               'ig': 'igbo',                      'id': 'indonesian',
            'ga': 'irish',                   'it': 'italian',                   'ja': 'japanese', 
            'jw': 'javanese',                'kn': 'kannada',                   'kk': 'kazakh', 
            'km': 'khmer',                   'ko': 'korean',                    'ku': 'kurdish (kurmanji)',      
            'ky': 'kyrgyz',                  'lo': 'lao',                       'la': 'latin', 
            'lv': 'latvian',                 'lt': 'lithuanian',                'lb': 'luxembourgish',
            'mk': 'macedonian',              'mg': 'malagasy',                  'ms': 'malay', 
            'ml': 'malayalam',               'mt': 'maltese',                   'mi': 'maori', 
            'mr': 'marathi',                 'mn': 'mongolian',                 'my': 'myanmar (burmese)',       
            'ne': 'nepali',                  'no': 'norwegian',                 'or': 'odia', 
            'ps': 'pashto',                  'fa': 'persian',                   'pl': 'polish', 
            'pt': 'portuguese',              'pa': 'punjabi',                   'ro': 'romanian', 
            'ru': 'russian',                 'sm': 'samoan',                    'gd': 'scots gaelic', 
            'sr': 'serbian',                 'st': 'sesotho',                   'sn': 'shona',
            'sd': 'sindhi',                  'si': 'sinhala',                   'sk': 'slovak', 
            'sl': 'slovenian',               'so': 'somali',                    'es': 'spanish',                 
            'su': 'sundanese',               'sw': 'swahili',                   'sv': 'swedish',
            'tg': 'tajik',                   'ta': 'tamil',                     'te': 'telugu', 
            'th': 'thai',                    'tr': 'turkish',                   'uk': 'ukrainian', 
            'ur': 'urdu',                    'ug': 'uyghur',                    'uz': 'uzbek', 
            'vi': 'vietnamese',              'cy': 'welsh',                     'xh': 'xhosa', 
            'yi': 'yiddish',                 'yo': 'yoruba',                    'zu': 'zulu',

    <HuggingFaceLM>: Callable [string with [mask]-> string]

   Possible BUG:
      que las probabilidades de la transformacion esten entre 0 y 1


    """
    def __init__(self, path:str='', transformation:dict={}, verbose=False):
        self.transformation = {'synonyms':0,'srcLang':None,'tgtLang':None,'repunctuation':False} #default Values
        for k in transformation: self.transformation[k] #raise error if the transformation doesn't exists 
        self.transformation.update(transformation)
        self.verbose=verbose


        if self.transformation['repunctuation']:
            self.doRepunctuation = txtRepunctuation()
        if self.transformation['synonyms']:
            self.doSynonyms = txtSynonyms(prob=self.transformation['synonyms'], aug_max=200)
        if self.transformation['tgtLang']:
            self.translator = BackTranslation()
        if self.verbose: print(self.transformation)


    def doBackTranslation(self, txt:str):
        sleep = 0
        try:
            txt = self.translator.translate(txt,src=self.transformation['srcLang'],tmp=self.transformation['tgtLang'], sleeping=sleep)
        except:
            sleep +=6
            print(f'googletranslate has blocked your request. Waiting: {sleep} secs')
            time.sleep(sleep)
            try:
               txt = self.translator.translate(txt,src=self.transformation['srcLang'],tmp=self.transformation['tgtLang'], sleeping=sleep)
            except:
                sleep *=10
                print(f'googletranslate has blocked your request. Waiting: {sleep} secs')
                time.sleep(sleep)
                try:
                    txt = self.translator.translate(txt,src=self.transformation['srcLang'],tmp=self.transformation['tgtLang'], sleeping=sleep)
                except:
                    raise RuntimeError('google translate has blocked your request, try again later.')

        return txt.result_text


    def __call__(self, data:pd.Series):
        out = []
        for sample in data:
            if self.transformation['repunctuation']: sample = self.doRepunctuation(sample)
            if self.transformation['synonyms']:      sample = self.doSynonyms(sample)
            if self.transformation['tgtLang']:       sample = self.doBackTranslation(sample)

            out.append(sample)
        return pd.Series(out)


if __name__ =='__main__':
    test = 2
    if test == 1:
        path='BETO/'
        data=pd.DataFrame({'src':['Hola este es un nuevo día','Unidad anatómica fundamental de todos los organismos vivos, generalmente microscópica, formada por citoplasma, uno o más núcleos y una membrana que la rodea']})
        transformation={'synonyms':0.1}
        augmentation = txtAugmentation(path,transformation)

        out = augmentation(data['src'])
        print(out[0])
        print(out[1])

    if test == 2:
        from BackTranslation import BackTranslation
        A = BackTranslation()
        q = A.translate('hola cómo estás hoy?', src='es',tmp='zh-cn')
        result = q.result_text
        print(result)
    