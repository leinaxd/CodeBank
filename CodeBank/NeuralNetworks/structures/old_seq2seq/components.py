
from typing import NamedTuple, Tuple, List
from torch.functional import Tensor
from torchtyping import TensorType
from glueFramework.framework import component, framework
from collections import namedtuple

from torchAPI.networkFramework.seq2seq.attentionModels import luongAttention
# from attentionModels import luongAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

#Todo:
# Ruptura de interferencia
#   -Separar componentes en funciones intercambiables 
#   -Solo puede haber una instancia por grupo
#   -No se pueden duplicar funciones.
#   -No se pueden retornar valores (o si con diccionario?)
#   -Definir ComponentsInterferenceError
# En vez de instanciar de generative models, hacerlo de un groupComponent particular
#Todo:
#   Puede el generative Model (component) ser dueño del wiring?
#   Pros
#   -Conoce exactamente el prototipo de las funciones
#   Cons
#   -Heredar del componente, hereda el wiring que esta mal
class generativeModel(component):
    """Base template for decoding step"""
    def __init__(self): pass
    # def on_init_loop(self, *args): """before decoding loop"""
    def on_init_attention(self, src_encodings, src_len): pass
    def on_init_IO_handler(self, *args): """initializations for input & output handler"""
    def on_init_step(self, prevOut, t, tgtEmb): pass
    def on_init_embed(self,tgtEmb): pass
    def on_end_step(self, prevOut): pass

    def on_prepare_input(self, t, prevOut, tgt): pass
    def on_prepare_values_keys(self,    values:     TensorType["batch","inSeq_length","enc_hidden", float], 
                                        keys:       TensorType[float],
                                        hypotheses: List[namedtuple]): return values, keys
    def on_decoding_step(self, *args): pass
    def on_attention_step(self, src_encodings, key, dec_hidden, enc_masks): pass
    def on_dropout_step(self, *args): pass
    def on_init_predictions(self): pass
    def on_eval_predictions(self, predictions): pass
    def on_finish_predictions(self, predictions): pass
    def on_break_loop(self, predictions): pass

class generativeFramework(framework, generativeModel):
    """wiring of generativeModel, inherit this to install plugins"""
    def decodingLoop(self): pass

class lstmStep(generativeModel):
    def __init__(self, input_size, hidden_size, device):
        self.decoder = nn.LSTMCell(input_size, hidden_size, device=device)

    def on_decoding_step(self, dec_input, dec_state):
        dec_state = self.decoder(dec_input, dec_state) #On step
        return dec_state

class attentionStep(generativeModel):
    def __init__(self, enc_hidden_size, dec_hidden_size, device):
        self.attention = luongAttention(enc_hidden_size, dec_hidden_size, device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)
                     # key: enc_hidden projected to dec_hidden for dot product
                     # e_t,i = (h_t^dec)^T W_h h_i^enc
    def on_init_attention(self, 
        src_hiddens:        TensorType['batch_size', 'src_len', 'enc_hidden_size'], 
        src_len:            TensorType['batch_size', int]):
        """Precalculates key vectors and values masks"""
        values = src_hiddens
        keys =  self.att_projection(src_hiddens)
        self.attention.generate_sent_masks(src_hiddens, src_len)
        return values, keys
    def on_attention_step(self, 
                            values: TensorType['batch_size', 'src_len', 'enc_hidden_size'], 
                            keys:   TensorType['batch_size', 'src_len', 'dec_hidden_size'], 
                            query:  TensorType['batch_size', 'dec_hidden_size']):
        prevOut = self.attention(values, keys, query) #on attention
        return prevOut

class dropoutStep(generativeModel):
    def __init__(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    def on_dropout_step(self, out):
        return self.dropout(out)


class gold_output_training(generativeModel):
    """Defines how to yield input, collect results and how to break the loop"""
    def __init__(self, embed_size, dec_hidden_size, output_size, padToken, device): 
        self.target_vocab_projection = nn.Linear(dec_hidden_size, output_size, False, device=device)
                 #P_t = softmax (W_vocab tanh(v_t))        
        self.device = device
        self.embed_size = embed_size
        self.prevOut:               TensorType['batch_size', 'embed_size'] = None
        self.predictions:   List[   TensorType['batch_size', 'embed_size']] = None
        self.decoding_time: int  = 0
        self.tgt_len:       int  = 0
        self.tgtEmb = nn.Embedding(output_size, embed_size, padToken, device = device)
        self.tgt:                   TensorType['tgt_len', 'batch_size', 'embed_size'] = None #Embedded target
    def on_init_IO_handler(self, tgt:TensorType['tgt_len','batch_size', int]):
        tgt = tgt[:-1] # Chop the <\s> token for max length sentences.
        self.tgt = self.tgtEmb(tgt)
        self.prevOut        = torch.zeros(self.tgt.shape[1], self.embed_size).to(self.device)
        self.predictions    = []
        self.tgt_len        = self.tgt.shape[0]
        self.decoding_time  = 0
    def on_break_loop(self):
        self.decoding_time += 1
        return self.decoding_time==self.tgt_len+1
    def on_prepare_input(self, dec_state) ->    TensorType['batch_size', 'dec_input_size']:
        true_last_word = self.tgt[self.decoding_time-1]      #TensorType['batch_size', 'embed_size']
        x    = torch.cat((true_last_word, self.prevOut), -1) #TensorType['batch_size', 'dec_input_size']
        return x, dec_state
    def on_prepare_values_keys(self, values, keys): #for attention
        return values, keys
    def on_eval_predictions(self, out):
        self.predictions.append(out)
        self.prevOut = out

    def on_finish_predictions(self) -> TensorType['tgt_len','batch_size','tgt_vocab', int]:
        embedPredictions = torch.stack(self.predictions) #(tgt_len, batch_size, dec_hidden_size)
        vocabPredictions = self.target_vocab_projection(embedPredictions) #From embed_size to Vocab_size
        logP = F.log_softmax(vocabPredictions, dim=-1)
        return logP #seq, batch and prob

class beam_search_evaluator(generativeModel):
    """Defines how to yield input, collect results and how to break the loop"""
    def __init__(self, beam_size, embed_size, dec_hidden_size, output_size, startToken, padToken, endToken, device):
        self.beam_size = beam_size
        self.startToken = startToken
        self.EOS = endToken
        self.dec_hidden_size = dec_hidden_size
        self.output_size = output_size
        self.device = device
        self.hypotheses: List[List[namedtuple]]               = None
        self.prevOut:    TensorType['beam_size','dec_hidden'] = None
        self.hyp_scores: TensorType['beam_size']              = None #pased Sentence prob.
        self.hyp_num = 1
        self.max_decoding_time = 70
        self.decodingTime = 0
        self.tgtEmb = nn.Embedding(output_size, embed_size, padToken, device = device)
        self.target_vocab_projection = nn.Linear(dec_hidden_size, output_size, False, device = device)
                 #P_t = softmax (W_vocab tanh(v_t))
        self.Hypothesis = namedtuple('Hypothesis', ['seq', 'score'])
    def on_init_IO_handler(self, _):
        self.decodingTime = self.max_decoding_time
        self.hypotheses = [[self.startToken]] #len = 1
        self.hyp_scores = torch.zeros(len(self.hypotheses), dtype=torch.float).to(self.device)
        self.prevOut    = torch.zeros(len(self.hypotheses), self.dec_hidden_size).to(self.device)
        self.prev_hypotheses = torch.zeros([1], dtype=torch.long).to(self.device)
        self.completed_hypotheses = []
    def on_break_loop(self):
        self.decodingTime -=1
        return True if not self.decodingTime or len(self.completed_hypotheses)==self.beam_size else False
    def on_prepare_input(self, dec_state) -> TensorType['nHypotheses', 'dec_input']:
        """1. cat (last word, prevOut) 2. expands keys and values"""
        self.hyp_num = len(self.hypotheses)
        #Decoder input: 
        # [beam_search predicted out[t-1]; out[t-1]]
        beamWord = [hyp[-1] for hyp in self.hypotheses] # De cada hipótesis, la última palabra
        beamWord = torch.tensor(beamWord, dtype=torch.long, device=self.device)
        beamWord = self.tgtEmb(beamWord)
        prevOut  = self.prevOut[self.prev_hypotheses]
        x = torch.cat([beamWord, prevOut], dim=-1)

        #Decoder state
        dec_hidden, dec_cell = dec_state 
        dec_state = (dec_hidden[self.prev_hypotheses], dec_cell[self.prev_hypotheses])
        return x, dec_state
    def on_prepare_values_keys(self, values, keys):
        """Instead of a batch of values, keys. Repeat same value, key for each hypotheses"""
        exp_values = values.expand(self.hyp_num, values.size(1), values.size(2))
        exp_keys   = keys.expand(self.hyp_num, keys.size(1), keys.size(2))
        return exp_values, exp_keys
    def on_eval_predictions(self, out: TensorType['nHypotheses', 'dec_hidden_size']):
        #next word score for each sentence
        logP = F.log_softmax(self.target_vocab_projection(out), dim=-1)
        #new sentences scores
        logP_cum = (self.hyp_scores.unsqueeze(1).expand_as(logP) + logP) #(nHypotheses x vocab)
        #choose best K words
        score, hyp_ix, word_ix = self._best_K_hypotheses(logP_cum)
        #updates best K hypotheses
        self._updateHypotheses(score, hyp_ix, word_ix)

        self.prevOut = out
    def on_finish_predictions(self):
        if len(self.completed_hypotheses) == 0:
            self.completed_hypotheses.append(self.Hypothesis(seq=self.hypotheses[0][1:],
                                                        score=self.hyp_scores[0].item()))
        #User expect most probable sentences first
        self.completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return self.completed_hypotheses
    
    def _best_K_hypotheses(self, logP_cum:TensorType['beam_size','vocab']):
        hyp_left = self.beam_size - len(self.completed_hypotheses)
        #Choose highest K words likelihoods from all hypotheses
        logP_cum = logP_cum.view(-1) #flatten
        hyp_prob, hyp_cand_ix = torch.topk(logP_cum, k=hyp_left)
        #Reshaped
        row = torch.div(hyp_cand_ix, self.output_size, rounding_mode='trunc') #hyp_cand_ix //self.output_size
        col = hyp_cand_ix % self.output_size
        return hyp_prob, row, col

    def _updateHypotheses(self, score, hyp_ix, word_ix):
        prev_hypotheses = [] #the hypothesis index
        new_hypotheses  = [] #The sequence so far
        new_hyp_scores  = [] #The sequence score
        for hyp_id, word, hyp_score in zip(hyp_ix, word_ix, score):
            hyp_id      = hyp_id.item()
            word        = word.item()
            hyp_score   = hyp_score.item()

            new_hyp = self.hypotheses[hyp_id] + [word]
            if word == self.EOS:
                self.completed_hypotheses.append(self.Hypothesis(seq=new_hyp[1:-1],
                                                                    score=hyp_score))
            else:
                prev_hypotheses.append(hyp_id)
                new_hypotheses.append(new_hyp)
                new_hyp_scores.append(hyp_score)
        self.prev_hypotheses = torch.tensor(prev_hypotheses, dtype=torch.long, device=self.device)
        self.hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
        self.hypotheses = new_hypotheses