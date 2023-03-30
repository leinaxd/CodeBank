
REQUIREMENTS = """
tokenizers
transformers
"""

from typing import List, Union


from transformers import AutoTokenizer #Pretrained tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
import tokenizers
# from transformers import AutoModel, AutoConfig, AutoTokenizer
# from transformers.modeling_outputs import SequenceClassifierOutput
# from transformers import logging

class Tokenizer:
    """
    Tokenizer __call__ form    
    """
    def __init__(self, tokenizer:tokenizers.Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data:Union[List, str]):
        if isinstance(data, str): data = [data]

        out = {'ids':[], 'type_ids':[],'attention_mask':[],'tokens':[],'offsets':[]}
        for sample in data:
            encoding = self.tokenizer.encode(sample)
            out['ids'].append(encoding.ids)
            out['type_ids'].append(encoding.type_ids)
            out['attention_mask'].append(encoding.attention_mask)
            out['tokens'].append(encoding.tokens)
            out['offsets'].append(encoding.offsets)
        return out
    

class Tokenizer_old:
    """
    Transforms a batch of sentences into a sequence of tokens

    Sources:
        https://huggingface.co/blog/how-to-train
    """

    def __init__(self, path='', algorithm=None, uncased=False):
        if path:
            self.model = AutoTokenizer.from_pretrained(path, do_lower_case=uncased)
        elif algorithm:
            self.model
        else:
            self.model = ByteLevelBPETokenizer()

    def __call__(self, batch:List[str]) -> List[int]:
        return self.model(batch)

    def train_from_files(self, paths:Union[List[str],str], vocab_size=30_000, min_frequency=2, show_progress=True, special_tokens=[]):
        tokenizer.train(data, vocab_size, min_frequency, special_tokens)
    def train_from_iterator(self, data:Union[List[str],str], vocab_size=30_000, min_frequency=2, show_progress=True, special_tokens=[],length=None):
        tokenizer.train_from_iterator(data, vocab_size, min_frequency, special_tokens,length)




if __name__ == '__main__':
    data = [
    "Schauspieler Orlando Bloom hat sich zur Trennung von seiner Frau , Topmodel Miranda Kerr , geäußert",
    "Actor Orlando Bloom announced his separation from his wife , supermodel Miranda Kerr"
            ]


    tokenizer = ByteLevelBPETokenizer()
    files = data
    special_tokens = ['<s>','<pad>','</s>','<ukn>','<mask>']
    

    # tokenizer.save_model('.','TEST_TOKENIZER')
    print(tokenizer.to_str())#.to_str()
    # tokenizer = AutoTokenizer

    # tokens = tokenizer(data)

    # print(tokens)



####################################
# DEPRECATED
####################################
def deprecated():
    import torch
    import random

    class MyTokenizer:#with sliding window
        def __init__(self, path, n_tokens, overlap:float, powerSW:bool=True, uncased=True, forceFullWindow=False, oneRandomBatch=False, lastBatch=False):
            self.window_size = n_tokens
            self.oneRandomBatch = oneRandomBatch
            self.lastBatch = lastBatch
            if lastBatch:   self.oneRandomBatch=False
            self.offset = int(overlap*self.window_size)
            if forceFullWindow:
                self.min_len = self.window_size-2 #at least do half window
            else:
                self.min_len = (self.window_size-2)*overlap #at least do half window
            self.power = powerSW #sliding window 

            self.tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=uncased)
            print(self.tokenizer.all_special_tokens)
            if '[CLS]' in self.tokenizer.all_special_tokens:
                self.bos = '[CLS]'
                self.pad = '[PAD]'
                self.eos = '[SEP]'
            else:
                self.bos = self.tokenizer.bos_token
                self.pad = self.tokenizer.pad_token
                self.eos = self.tokenizer.eos_token

        def slidingWindow(self, sample:str, window_size) -> List[str]:
            if not self.power: return [sample[:window_size]] #force a truncation here
            result = []
            s=0
            if len(sample) <= self.min_len: result.append(sample) #for texts smaller than the window
            while s < len(sample)-self.min_len:
                e = s + window_size
                result.append(sample[s:e])
                s = e-self.offset
            return result

        def test_1(self):
            print('long text example')
            txt = '1 2 3 4 5 6 7 8 9 10 11 12 13 14'
            print(f"expected: {self.tokenizer(txt)}")
            self(txt)
            print(f"obtained: {self(txt)}")
        def test_2(self):
            print('batched test')
            txt = ['Hola este es el comienzo', ' de un texto','corto']
            print(f"expected: {self.tokenizer(txt, return_tensors='pt', padding=True)}")
            print(f"obtained: {self(txt[0])}")
            print(f"obtained: {self(txt[1])}")
            print(f"obtained: {self(txt[2])}")
        def tokenize(self, txt:str):
            return self.tokenizer.tokenize(txt)
        def __call__(self, txt:str, label:int = None) -> List[list]:
            if isinstance(txt, list): 
                features = self.tokenizer(txt, padding=True, truncation=True, max_length=self.window_size, return_tensors='pt')
                if label is not None: features['labels'] = torch.tensor([label])
                return features
            tokens = self.tokenizer.tokenize(txt)
            batch_tokens = self.slidingWindow(tokens, self.window_size-2) #without bos and eos
            if self.oneRandomBatch: batch_tokens = [random.choice(batch_tokens)]
            if self.lastBatch:      batch_tokens = [batch_tokens[-1]]
                
            input_ids, input_mask, token_type = self.addSpecialTokens(batch_tokens)
            
            # input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # reconstructed = self.tokenizer.convert_tokens_to_string(tokens)
            features = {
            'input_ids': torch.tensor(input_ids, dtype=torch.int),
            'attention_mask': torch.tensor(input_mask, dtype=torch.int),
            'token_type_ids': torch.tensor(token_type, dtype=torch.int)
            }
            if label is not None: features['labels'] = torch.tensor([label]*len(input_ids)) #warning with label 0
            return features

        def addSpecialTokens(self, batch_tokens:List[List[str]]) -> List[str]:
            #left side information, right padding
            input_ids = []
            input_mask = []
            token_type_ids = []
            maxLen = max([len(sample) for sample in batch_tokens])+2
            for tok_sample in batch_tokens:
                tokens = [self.bos] + tok_sample + [self.eos]
                input_mask.append([1]*len(tokens) + [0]*(maxLen-len(tokens)))
                tokens = tokens+[self.pad]*(maxLen-len(tokens)) 
                input_ids.append( self.tokenizer.convert_tokens_to_ids(tokens) )
                token_type_ids.append([0]*maxLen )
            return input_ids, input_mask, token_type_ids
            
        def untokenize(self, features):
            input_ids = features['input_ids'].tolist()
            txt = []
            for id_sample in input_ids:
                tokens = self.tokenizer.convert_ids_to_tokens(id_sample[1:-1])#ignore bos, eos
                txt.append( self.tokenizer.convert_tokens_to_string(tokens) )
            return txt

    #Old tokenizer

    # tokenizer = MyTokenizer(path, 20, 0, True)
    # tokenizer.test_1()
    # tokenizer.test_2()
    # input = tokenizer('A B C D E F G H I J K',1)

    # model.loss(input)
    #if test==2:
    # n_tokens=200
    # overlap=0
    # power=True
    # force=True
    # tokenizer = MyTokenizer(path, n_tokens, overlap, power,forceFullWindow=force)
    # pos = 1
    # for i, sample in enumerate(train_dataset['transcription']):
    #   if i!=pos: continue
    #   input = tokenizer(sample)
    #   rec = tokenizer.untokenize(input)
    #   print(rec)
    #   print(sample,"\n")