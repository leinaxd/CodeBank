
import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging
from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRandomSampling
from CodeBank.DatasetFramework.DataAugmentation.transformations import txtRepunctuation

from BackTranslation import BackTranslation
# from CodeBank.Math.Random import choices

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
            'af': 'afrikaans',               'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque',
            'be': 'belarusian',              'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa',
            'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish',
            'nl': 'dutch',                   'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian',
            'gl': 'galician',                'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian',
            'iw': 'hebrew',                  'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'id': 'indonesian',
            'ga': 'irish',                   'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean',
            'ku': 'kurdish (kurmanji)',      'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish',
            'mk': 'macedonian',              'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian',
            'my': 'myanmar (burmese)',       'ne': 'nepali', 'no': 'norwegian', 'or': 'odia', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese',
            'pa': 'punjabi',                 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona',
            'sd': 'sindhi',                  'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali',
            'es': 'spanish',                 'su': 'sundanese',
            'sw': 'swahili',                 'sv': 'swedish',
            'tg': 'tajik',                   'ta': 'tamil',
            'te': 'telugu',                  'th': 'thai',
            'tr': 'turkish',                 'uk': 'ukrainian', 'ur': 'urdu',
            'ug': 'uyghur',                  'uz': 'uzbek', 'vi': 'vietnamese',
            'cy': 'welsh',                   'xh': 'xhosa', 'yi': 'yiddish',
            'yo': 'yoruba',                  'zu': 'zulu',

    <HuggingFaceLM>: Callable [string with [mask]-> string]

   Possible BUG:
      que las probabilidades de la transformacion esten entre 0 y 1


    """
    def __init__(self, path:str, transformation:dict, verbose=False, doSoftmax=False, sleep=0):
        self.transformation = {'synonyms':0,'srcLang':None,'tgtLang':None,'repunctuation':False} #default Values
        self.sleep=sleep
        self.transformation.update(transformation)

        if self.transformation['synonyms']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            logging.set_verbosity_error() #Ignore unused weights warning. (this model is for finetunning)
            self.model = AutoModelForMaskedLM.from_pretrained(path)
            logging.set_verbosity_warning()
            self.model.to(self.device)
            # self.model.config.max_position_embeddings = 512
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.sampling = txtRandomSampling(self.transformation['synonyms'], maskToken=self.tokenizer.mask_token)

        # self.selection = choices('softmax')
        self.verbose=verbose
        self.count  = 0
        self.max_length=512
        self.doSoftmax=doSoftmax

        if self.transformation['tgtLang']:
            self.translator = BackTranslation()
        if self.transformation['repunctuation']:
            self.doRepunctuation = txtRepunctuation()
        if self.verbose: print(self.transformation)
    # def __init__(self, transformations:dict, HuggingFaceLM:str, maskToken:str):

    #     self.sampling = txtRandomSampling(transformations['synonyms'], maskToken)
    #     self.modelLM = HuggingFaceLM

    def doSynonyms(self, txt:str):
        if self.verbose: 
            self.count+=1
            print(f"{self.count}",end='')
        input_txt = self.sampling(txt)
        input = self.tokenizer(input_txt, return_tensors = 'pt', truncation=True,max_length=self.max_length)
        input = {k:v.to(self.device) for k, v in input.items()}

        input_ids = input['input_ids'][0] #ignore batching dim
        masked_ix, = torch.where(input_ids == self.tokenizer.mask_token_id)

        self.model.eval()
        predictions = self.model(**input).logits

        for ix in masked_ix.tolist():
            # predicted_token = self.selection(predictions[0,ix])
            mask_prob = predictions[0,ix]
            if self.doSoftmax: mask_prob = torch.nn.functional.softmax(mask_prob, dim=-1)
            predicted_token = torch.multinomial(mask_prob, 1)
            input_ids[ix] = predicted_token

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:-1])
        txt = self.tokenizer.convert_tokens_to_string(tokens)
        return txt

    def doBackTranslation(self, txt:str):
        sleep = self.sleep
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
    test = 1
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
