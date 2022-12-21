import time
from BackTranslation import BackTranslation
import nlpaug.augmenter.word as naw
import torch

class txtBackTranslation:
    """
    !pip install BackTranslation
    sources:
        https://github.com/hhhwwwuuu/BackTranslation
    Languages:
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
    
    Also:
        #Citar traductor
        Helsinki-NLP/opus-mt-es-en
        Helsinki-NLP/opus-mt-en-es
        Helsinki-NLP/opus-mt-es-de
        Helsinki-NLP/opus-mt-de-es
    """

    def __init__(self, srcLang:str, tgtLang:str):
        if len(srcLang) <= 4 and len(srcLang)<=4:
            self.srcLang = srcLang
            self.tgtLang = tgtLang
            self.model = BackTranslation()
            self.call = self.doBackTranslation_1
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = naw.BackTranslationAug(srcLang, tgtLang, max_length=512, device=device)
            self.call  = self.doBackTranslation_2
    def __call__(self, txt:str): return self.call(txt)
    
    def doBackTranslation_2(self, txt:str):
        print(isinstance(txt,str))
        out = self.model.augment(txt)
        print(out[0])
        if isinstance(txt,str): return out[0]
        else:                   return out

    def doBackTranslation_1(self, txt:str):
        sleep = 0
        try:
            txt = self.translator.translate(txt,src=self.srcLang,tmp=self.tgtLang, sleeping=sleep)
        except:
            sleep +=6
            print(f'googletranslate has blocked your request. Waiting: {sleep} secs')
            try:
                time.sleep(sleep)
                txt = self.translator.translate(txt,src=self.transformation['srcLang'],tmp=self.transformation['tgtLang'], sleeping=sleep)
            except:
                sleep *=10
                print(f'googletranslate has blocked your request. Waiting: {sleep} secs')
                try:
                    time.sleep(sleep)
                    txt = self.translator.translate(txt,src=self.transformation['srcLang'],tmp=self.transformation['tgtLang'], sleeping=sleep)
                except:
                    raise RuntimeError('google translate has blocked your request, try again tomorrow.')

        return txt.result_text




if __name__ == '__main__':
    corpus_1 =     original = """Cumpliendo con mi oficio piedra con piedra, pluma a pluma, pasa el invierno y deja sitios abandonados, habitaciones muertas: yo trabajo y trabajo, debo substituir tantos olvidos, llenar de pan las tinieblas, fundar otra vez la esperanza."""
    test = 2
    if test == 1:
        A = BackTranslation()
        q = A.translate('hola cómo estás hoy?', src='es',tmp='zh-cn')
        result = q.result_text
        print(result)
    if test == 2:
        src = 'Helsinki-NLP/opus-mt-es-de'
        tgt = 'Helsinki-NLP/opus-mt-de-es'

        aug = txtBackTranslation(src,tgt)
        out = aug(corpus_1)
        print(out)
    