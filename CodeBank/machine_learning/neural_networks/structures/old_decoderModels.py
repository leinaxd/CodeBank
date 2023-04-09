#Author: Eichenbaum, Daniel
#Decoders Library
#   Includes:
#       LSTMDecoder
#   Features:
#       attention
#       beam_search
from torchtyping import TensorType
from typing import Tuple, List, Union
from collections import namedtuple
# from glueFramework import framework, component
from torchAPI.networkFramework.seq2seq.attentionModels import luongAttention, myAttention_ver_1
from torchAPI.networkFramework.seq2seq.components import generativeModel, lstmStep, attentionStep, dropoutStep, gold_output_training, beam_search_evaluator
from torchAPI.networkFramework.seq2seq.recurrentUnits import resRNN, resRNN_v2

import torch
import torch.nn as nn
import torch.nn.functional as F




class myDecoder(generativeModel):
    """LSTM Decoder with attention [Optional beam search for eval]"""
    def __init__(self, embed_size: int, dec_hidden_size: int, output_size:int, dropout_rate: float, startToken, padToken, endToken, device, evalMode):
        enc_hidden_size = 2*embed_size
        dec_hidden_size = embed_size
        super().__init__(generativeModel)
        self.glue(lstmStep(embed_size+dec_hidden_size, dec_hidden_size, device = device))
        self.glue(attentionStep(enc_hidden_size, dec_hidden_size, device = device))
        self.glue(dropoutStep(dropout_rate))
        if evalMode:    self.glue( beam_search_evaluator(10, embed_size, dec_hidden_size, output_size, startToken, padToken, endToken, device) )
        else:           self.glue( gold_output_training(embed_size, dec_hidden_size, output_size, padToken, device) )
        # self.glue(beam_search_training(dec_hidden_size, output_size, device))
        self.compile()
        self.hidden_size = dec_hidden_size
        self.EOS = endToken
        self.device = device
        
    def decodingLoop(self,
                dec_state:  Tuple[  TensorType["batch_size","dec_hidden_size",float], 
                                    TensorType["batch_size","dec_hidden_size",float]], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                src_len:            TensorType["batch_size", int],
                tgt:                TensorType["tgt_len","batch_size", int]
                ) ->                TensorType['tgt_len','batch_size','tgt_vocab', int]:
        self.on_init_IO_handler(tgt)
        values, keys = self.on_init_attention(enc_hiddens, src_len)#Todos los args de input
        # t = 0
        while True:
            # t = t+1
            # if t==5: break
            if self.on_break_loop(): break
            dec_input, dec_state    = self.on_prepare_input(dec_state)
            dec_state               = self.on_decoding_step(dec_input, dec_state)
            bValues, bKeys          = self.on_prepare_values_keys(values, keys) #Batch values/keys if needed
            out                     = self.on_attention_step(bValues, bKeys, dec_state[0])
            out                     = self.on_dropout_step(out)#u otra regularizacion (ej. +W'W)
            self.on_eval_predictions(out)
        return self.on_finish_predictions() #sequence and/or Probability for each batch


      
class myDecoder_expandedVersion(nn.Module):
    """A LSTM decoder step with attention"""
    def __init__(self, embed_size: int, dec_hidden_size: int, output_size:int, dropout_rate: float, startToken, padToken, endToken, device, evalMode):
        super().__init__()
        enc_hidden_size = 2*embed_size
        dec_hidden_size = embed_size

        self.hidden_size = dec_hidden_size
        self.embed_size = embed_size
        self.EOS = endToken
        self.device = device

        self.decoder = nn.LSTMCell(embed_size+dec_hidden_size, dec_hidden_size, device=device)
        self.attention = luongAttention(enc_hidden_size, dec_hidden_size, device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)
        self.dropout = nn.Dropout(dropout_rate)      
        self.target_vocab_projection = nn.Linear(dec_hidden_size, output_size, False, device=device)
        self.prevOut:               TensorType['batch_size', 'embed_size'] = None
        self.predictions:   List[   TensorType['batch_size', 'embed_size']] = None
        self.decoding_time: int  = 0
        self.tgt_len:       int  = 0
        self.tgtEmb = nn.Embedding(output_size, embed_size, padToken, device = device)
        self.tgt:                   TensorType['tgt_len', 'batch_size', 'embed_size'] = None #Embedded target


    def decodingLoop(self,
                dec_state:  Tuple[  TensorType["batch_size","dec_hidden_size",float], 
                                    TensorType["batch_size","dec_hidden_size",float]], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                src_len:            TensorType["batch_size", int],
                tgt:                TensorType["tgt_len","batch_size", int]
                ) ->                TensorType['tgt_len','batch_size','tgt_vocab', int]:
        tgt = tgt[:-1]
        self.tgt = self.tgtEmb(tgt)
        self.prevOut        = torch.zeros(self.tgt.shape[1], self.embed_size).to(self.device)
        self.predictions    = []
        self.tgt_len        = self.tgt.shape[0]
        self.decoding_time  = 0

        values = enc_hiddens
        keys =  self.att_projection(enc_hiddens)
        self.attention.generate_sent_masks(enc_hiddens, src_len)
        while True:
            self.decoding_time += 1
            if self.decoding_time==self.tgt_len+1: break

            true_last_word = self.tgt[self.decoding_time-1]      #TensorType['batch_size', 'embed_size']
            dec_input      = torch.cat((true_last_word, self.prevOut), -1) #TensorType['batch_size', 'dec_input_size']

            dec_state = self.decoder(dec_input, dec_state)

            bValues, bKeys          =  values, keys

            out = self.attention(bValues, bKeys, dec_state[0]) #on attention
            out = self.dropout(out)

            self.predictions.append(out)
            self.prevOut = out

        embedPredictions = torch.stack(self.predictions) #(tgt_len, batch_size, dec_hidden_size)
        vocabPredictions = self.target_vocab_projection(embedPredictions) #From embed_size to Vocab_size
        logP = F.log_softmax(vocabPredictions, dim=-1)
        return logP





class beam_search_decoder(nn.Module):
    def __init__(self, embed_size: int, dec_hidden_size: int, output_size:int, dropout_rate: float, startToken, padToken, endToken, device, evalMode):
        super().__init__()
        enc_hidden_size = 2*embed_size
        dec_hidden_size = embed_size
        self.hidden_size = dec_hidden_size
        self.output_size = output_size
        self.EOS = endToken
        self.device = device
        self.beam_size = 10
        self.startToken = startToken
        self.dec_hidden_size = dec_hidden_size

        self.decoder = nn.LSTMCell(embed_size+dec_hidden_size, dec_hidden_size, device=device)
        self.attention = luongAttention(enc_hidden_size, dec_hidden_size, device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)
        self.dropout = nn.Dropout(dropout_rate)

        self.prevOut:    TensorType['beam_size','dec_hidden'] = None
        self.hyp_scores: TensorType['beam_size']              = None #pased Sentence prob.
        self.hyp_num = 1
        self.decodingTime = 0
        self.max_decoding_time = 70
        self.Hypothesis = namedtuple('Hypothesis', ['seq', 'score'])
        self.hypotheses: List[List[namedtuple]]               = None
        self.target_vocab_projection = nn.Linear(dec_hidden_size, output_size, False, device = device)
        self.tgtEmb = nn.Embedding(output_size, embed_size, padToken, device = device)
        

    def decodingLoop(self,
                dec_state:  Tuple[  TensorType["batch_size","dec_hidden_size",float], 
                                    TensorType["batch_size","dec_hidden_size",float]], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                src_len:            TensorType["batch_size", int],
                tgt:                TensorType["tgt_len","batch_size", int]
                ) ->                TensorType['tgt_len','batch_size','tgt_vocab', int]:
        ### on_init_IO_handled
        self.decodingTime = self.max_decoding_time
        self.hypotheses = [[self.startToken]] #len = 1
        self.hyp_scores = torch.zeros(len(self.hypotheses), dtype=torch.float).to(self.device)
        self.prevOut    = torch.zeros(len(self.hypotheses), self.dec_hidden_size).to(self.device)
        self.prev_hypotheses = torch.zeros([1], dtype=torch.long).to(self.device)
        self.completed_hypotheses = []

        ### on_init_attention
        values = enc_hiddens
        keys =  self.att_projection(enc_hiddens)
        self.attention.generate_sent_masks(enc_hiddens, src_len)
        ###
        while True:
            #breaking condition
            self.decodingTime -=1
            break_cond = True if not self.decodingTime or len(self.completed_hypotheses)==self.beam_size else False
            if break_cond: break
            #on_prepare_input(dec_state)
            self.hyp_num = len(self.hypotheses)
            beamWord = [hyp[-1] for hyp in self.hypotheses] # De cada hipótesis, la última palabra
            beamWord = torch.tensor(beamWord, dtype=torch.long, device=self.device)
            beamWord = self.tgtEmb(beamWord)
            prevOut  = self.prevOut[self.prev_hypotheses]
            dec_input = torch.cat([beamWord, prevOut], dim=-1)
            #Decoder state
            dec_hidden, dec_cell = dec_state 
            dec_state = (dec_hidden[self.prev_hypotheses], dec_cell[self.prev_hypotheses])
            #on_decoding_step
            dec_state = self.decoder(dec_input, dec_state)
            #on_prepare_values_keys(values, keys)
            bValues = values.expand(self.hyp_num, values.size(1), values.size(2))
            bKeys   = keys.expand(self.hyp_num, keys.size(1), keys.size(2))
            #on_attention_step()
            out = self.attention(bValues, bKeys, dec_state[0])
            out = self.dropout(out)
            #on_eval_predictions
            logP = F.log_softmax(self.target_vocab_projection(out), dim=-1)
            logP_cum = (self.hyp_scores.unsqueeze(1).expand_as(logP) + logP) #(nHypotheses x vocab)
            score, hyp_ix, word_ix = self._best_K_hypotheses(logP_cum)
            self._updateHypotheses(score, hyp_ix, word_ix)
            self.prevOut = out
        #On_finish_predictions()
        if len(self.completed_hypotheses) == 0:
            self.completed_hypotheses.append(self.Hypothesis(seq=self.hypotheses[0][1:], score=self.hyp_scores[0].item()))
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


class beam_search_decoder2:
    def __init__(self, embed_size: int, hidden_size: int, output_size: int, dropout_rate: float, startToken, padToken, endToken):
        super().__init__(embed_size, hidden_size, endToken)
        self.tgtEmb = nn.Embedding(output_size, embed_size, padToken)
        self.dropout = nn.Dropout(dropout_rate)
        self.target_vocab_projection = nn.Linear(embed_size, output_size, False) #P_t = softmax (W_vocab tanh(v_t))
        ###########
        # Parametros:
        ################
        self.max_decoding_time_step=70
        self.beam_size=10
        self.SOS = startToken #self.loader.vocab.tgt['<s>']
        self.EOS = endToken
        self.output_size = output_size

        self.Hypothesis = namedtuple('Hypothesis', ['candidate', 'score'])  #Obs. Sentences = Hypotheses
    def decodingLoop(self, src_encodings) -> List[namedtuple]: #Todo: Return tensorType['batch', 'outSeq_length']
        """ Given a single source sentence, perform an heuristic search (greedy, BFS, beam_search, DFS, Goal Tree) """
        #Todo:
        #   Requiere:
        #       -Candidatos
        #       -scoreFn: Puntaje por cada candidato
        #   Devuelve:
        #       -Secuencia
        # key = self.att_projection(src_encodings)  #Precalculo W_att · h_i
        # prevOut = torch.zeros(1, self.hidden_size, device=self.device)
        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device) #1 = len(hypotheses)
        hypotheses = [[self.SOS]] #List[List[Vocab]] current generation state, (candidate, score)
        completed_hypotheses: List[namedtuple] = []

        for _ in range(self.max_decoding_time_step):
            if len(completed_hypotheses) == self.beam_size: break
            
            x, exp_src_encodings, exp_key                   = self.on_prepare_input(prevOut, hypotheses, src_encodings, key)
            # prevOut, dec_state                              = self.on_step(x, dec_state, exp_src_encodings, exp_key)
            logP, out, O_t                                  = self.on_eval(prevOut)           
            prev_hyp_ix, new_word_ix, hyp_cand              = self.on_choose_k(logP, completed_hypotheses, hyp_scores)
            prev_hypotheses, hypotheses, hyp_scores         = self.on_update_hypotheses(prev_hyp_ix, new_word_ix, hyp_cand, completed_hypotheses)
            dec_state, prevOut                              = self.on_update(prev_hypotheses, dec_state, O_t)

        return self.on_end(completed_hypotheses)
        #Actualmente Tengo:
        #   k hypotheses con k hyp_scores
        #Si genero </s>:
        #   añado hypothesis j a completed_hypotheses

    def on_eval(self, prevOut):
        #get next prediction
        #   Input:
        #       inputWord: Palabra anterior para cada hipótesis
        #       att_out: Representación de la oración de entrada
        #       dec_state: init_state
        #   Output:
        #       logP: Prediction for each hipotheses
        O_t = self.dropout(prevOut)
        out = self.target_vocab_projection(O_t) #dim(O_t) = embed_size, dim(out) = |V_tgt|
        logP = F.log_softmax(out, dim=-1)
        return logP, out, O_t
    def on_choose_k(self, logP, completed_hypotheses, hyp_scores):
        #Recalculate the likelihood of each hipothesis (including new word)
        #   Input: 
        #       logP: likelihood of any new word P(w_n)
        #   Output:
        #       logP_cum: likelihood of sentences with any new word P(w_1..w_n)
        hyp_left = self.beam_size - len(completed_hypotheses)
        logP_cum = (hyp_scores.unsqueeze(1).expand_as(logP) + logP).view(-1) 
        #Choose best K new words for each hypotheses
        #   Input:
        #       logP_cum: probability of choosing any new word for each sentence
        #   Output:
        #       hyp_cand: top K words for each sentence
        hyp_cand, hyp_cand_ix = torch.topk(logP_cum, k=hyp_left)
        #Que hace esto?
        prev_hyp_ix = hyp_cand_ix // self.output_size
        new_word_ix = hyp_cand_ix % self.output_size
        # hyp_cand_ix = prev_hyp_ids*output_size + hyp_word_ids
        prev_hyp_ix = [h.item() for h in prev_hyp_ix]
        new_word_ix = [h.item() for h in new_word_ix]
        hyp_cand    = [h.item() for h in hyp_cand]
        return prev_hyp_ix, new_word_ix, hyp_cand
    
    def on_update_hypotheses(self, prev_hyp_ix, new_word_ix, hyp_cand, completed_hypotheses:List[namedtuple]):
        #
        prev_hypotheses = []
        new_hypotheses  = []
        new_hyp_scores  = []
        for prev_hyp, new_word, hyp_score in zip(prev_hyp_ix, new_word_ix, hyp_cand):
            new_hyp  = [prev_hyp] + [new_word]

            if new_word == self.EOS:
                completed_hypotheses.append(self.Hypothesis(candidate=new_hyp[1:-1],
                                                        score=hyp_score))
            else:
                prev_hypotheses.append(prev_hyp)
                new_hypotheses.append(new_hyp)
                new_hyp_scores.append(hyp_score)
        prev_hypotheses = torch.tensor(prev_hypotheses, dtype=torch.long, device=self.device)
        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
        return prev_hypotheses, hypotheses, hyp_scores

    def on_update(self, prev_hypotheses, dec_state, O_t):
        dec_hidden, dec_cell = dec_state #Todo dec_state[prev_hypotheses]
        dec_state = (dec_hidden[prev_hypotheses], dec_cell[prev_hypotheses])
        prevOut = O_t[prev_hypotheses]
        return dec_state, prevOut

    def on_end(self, completed_hypotheses: List[namedtuple]):
        #if You didn't finish any hipothesis, just append what you got
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(self.Hypothesis(candidate=self.hypotheses[0][1:],
                                                   score=self.hyp_scores[0].item()))
        #User expect most probable sentences first
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses


#############
# Propio
#Ideas de por que no funciona:
#   1. -falta de consistencia (chequeo por embedding o por vocab, pero no una mezcla)
#   2. -faltan parametros
#   3. -
class myDecoder_ver_1(nn.Module):
    """A LSTM decoder step with attention"""
    def __init__(self, embed_size: int, enc_hidden_size:int, dec_hidden_size: int, output_size:int, 
                    nStacks:int=1, nDepth:int=1, recurrentUnit='LSTM', residual:bool=True, 
                    attType:str = 'dot',
                    dropout_rate: float=0, 
                    endToken=None, padToken=None, 
                    seqEmb:nn.Embedding=None, device=None):
        super().__init__()
        self.device = device
        # self.EOS = torch.tensor(endToken, device = self.device)
        self.EOS = endToken[0] #just the number, not list
        self.maxDecodingTime = 50

        self.embed_size = embed_size

        if recurrentUnit == 'LSTM':
            self.decoderUnit  = resRNN(embed_size,dec_hidden_size, nDepth, nStacks, residual,residual, device=device)
        elif recurrentUnit == 'resRNN':
            self.decoderUnit  = resRNN_v2(embed_size,dec_hidden_size, nDepth, nStacks, residual,residual, device=device)
        else: raise Exception(f"recurrentUnit {recurrentUnit} unknow, try 'LSTM' or 'resRNN'")

        self.layerOutput    = nn.Linear(dec_hidden_size, embed_size, False, device=device) #Same for all layers

        
        
        self.attention      = myAttention_ver_1(attType, enc_hidden_size, dec_hidden_size,device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)

        self.outNN          = nn.Linear(2*enc_hidden_size, embed_size, False, device=device)
        self.outDropout     = nn.Dropout(dropout_rate)
        self.vocab_projection = nn.Linear(embed_size, output_size, False, device=device)
        # self.seqEmb = seqEmb
        self.seqEmb = nn.Embedding(output_size, embed_size, padToken[0])
    def forward(self,
                dec_initial_state:  TensorType['nMem','nStacks','batch_size','dec_hidden_size',float], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                enc_len:            TensorType["batch_size", int],
                tgt:                Tuple[  TensorType['batch_size','tgt_len',int],   
                                            TensorType['batch_size',int]]
                ) ->                TensorType['batch_size','tgt_len','tgt_vocab', int]:
        tgt = tgt[0] if self.training else None #Precalc emb

        # dec_state = (dec_initial_state[0].squeeze(0), dec_initial_state[1].squeeze(0)) #delete num_layers dim
        # dec_states = {i: (dec_initial_state[0].squeeze(0),dec_initial_state[1].squeeze(0)) for i in range(self.nLayers)}
        # self.decoderUnit.initialize(dec_initial_state)
        dec_state = dec_initial_state
        # dec_state = (dec_initial_state[0].squeeze(0),dec_initial_state[1].squeeze(0)) #delete num_layers dim

        decoding_time  = 0
        batch_size     = enc_len.shape[0]
        prevOut        = torch.zeros(batch_size, self.embed_size, device=self.device)
        tgt_len        = torch.zeros(batch_size, dtype=int, device=self.device)
        predictions    = []
        flags          = torch.tensor([False]*batch_size, device=self.device) #Flags when reached <end>

        values = enc_hiddens
        keys =  self.att_projection(enc_hiddens)
        self.attention.generate_sent_masks(enc_hiddens, enc_len)
        while decoding_time < self.maxDecodingTime:

            # decOut = self.decoderUnit(prevOut)
            decOut, dec_state = self.decoderUnit(prevOut, dec_state)
            # dec_state = (decOut, decCell)

            c_t = self.attention(values, keys, decOut)
    
            #Combining Input representation and Decoder state
            combined_output = torch.cat((decOut, c_t), 1)    
            
            #Producing output from combined representation
            decOut = self.outNN(combined_output) #From [dec_hidden; c_t] to embed_size
            decOut = self.outDropout(decOut)

            predictions.append(decOut)  #Por falta de ram, solo guardo el embedding
            vocabPrediction = self.vocab_projection(decOut)
            symbol = self.greedyStrategy(vocabPrediction)

            #Forcing golden words? It overwrites tgt_len, so it won't learn this parameter
            # if self.training: symbol = tgt[:,decoding_time]  #TensorType['batch_size']
            prevOut = self.seqEmb(symbol)
            decoding_time += 1

            #Break when prediction has all <EOS>
            ixEOS = self.EOS == symbol
            flags = torch.logical_or(flags, ixEOS)
            tgt_len[ixEOS] = decoding_time
            if torch.all(flags): break
        tgt_len[torch.logical_not(ixEOS)] = decoding_time
        embedPredictions = torch.stack(predictions,1) #(batch_size, tgt_len, vocab_size)
        vocabPredictions = self.vocab_projection(embedPredictions) #From embed_size to Vocab_size
        logP = F.log_softmax(vocabPredictions,dim=2) #ouput = probability of each word
        return logP, tgt_len

    def greedyStrategy(self, utility: TensorType['batch_size','vocab_size']
                            )   ->    TensorType['batch_size',int]:
        """Just take the best option every time"""
        return torch.max(utility, dim=1).indices

class myDecoder_ver_2(nn.Module):
    """A LSTM decoder step with attention"""
    def __init__(self, embed_size: int, enc_hidden_size:int, dec_hidden_size: int, output_size:int, dropout_rate: float, endToken, seqEmb:nn.Embedding, device):
        super().__init__()
        self.device = device
        # self.EOS = torch.tensor(endToken, device = self.device)
        self.EOS = endToken
        self.maxDecodingTime = 100
        self.embed_size = embed_size

        self.decoder        = nn.LSTMCell(embed_size, dec_hidden_size, device=device)
        self.attention      = myAttention_ver_1(device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)
        self.combOutput     = nn.Linear(2*enc_hidden_size, output_size, False, device=device)
        self.dropout        = nn.Dropout(dropout_rate)      
        self.vocab_projection = nn.Linear(embed_size, output_size, False, device=device)
        self.seqEmb = seqEmb
    def forward(self,
                dec_initial_state:  Tuple[  TensorType['num_layers','batch_size','dec_hidden_size',float], 
                                            TensorType['num_layers','batch_size','dec_hidden_size',float]], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                enc_len:            TensorType["batch_size", int],
                tgt:                Tuple[  TensorType['batch_size','tgt_len',int],   
                                            TensorType['batch_size',int]]
                ) ->                TensorType['batch_size','tgt_len','tgt_vocab', int]:
        tgt = tgt[0] if self.training else None #Precalc emb

        dec_state = (dec_initial_state[0].squeeze(0),dec_initial_state[1].squeeze(0)) #delete num_layers dim

        decoding_time  = 0
        batch_size     = enc_len.shape[0]
        prevOut        = torch.zeros(batch_size, self.embed_size, device=self.device)
        tgt_len        = torch.zeros(batch_size, dtype=int)
        predictions    = []
        flags          = torch.tensor([False]*batch_size, device=self.device) #Flags when reached <end>

        values = enc_hiddens
        keys =  self.att_projection(enc_hiddens)
        self.attention.generate_sent_masks(enc_hiddens, enc_len)
        while decoding_time < self.maxDecodingTime:
            dec_input = prevOut  #TensorType['batch_size', 'embed_size']

            dec_state = self.decoder(dec_input, dec_state)
            c_t = self.attention(values, keys, dec_state[0])
    
            #Combining Input representation and Decoder state
            combined_output = torch.cat((dec_state[0], c_t), 1)    
            
            #Producing output from combined representation
            decOut = self.combOutput(combined_output) #From [dec_hidden; c_t] to vocab_size
            decOut = torch.tanh(decOut)
            decOut = self.dropout(decOut)

            predictions.append(decOut) 
            symbol = self.greedyStrategy(decOut)

            #Forcing golden words
            if self.training: symbol = tgt[:,decoding_time]  #TensorType['batch_size']
            prevOut = self.seqEmb(symbol)
            decoding_time += 1

            #Break when prediction has all <EOS>
            ixEOS = self.EOS == symbol
            flags = torch.logical_or(flags, ixEOS)
            tgt_len[ixEOS] = decoding_time
            if torch.all(flags): break
        
        tgt_len[torch.logical_not(ixEOS)] = decoding_time #Saves final length
        logP = torch.stack(predictions,1) #(batch_size, tgt_len, vocab_size)
        logP = F.log_softmax(logP,dim=2) #ouput = probability of each word
        return logP, tgt_len

    def greedyStrategy(self, utility: TensorType['batch_size','vocab_size']
                            )   ->    TensorType['batch_size',int]:
        """Just take the best option every time"""
        return torch.max(utility, dim=1).indices


#TODO: Ver2 predicciones sin probabilidad (creo que es peor, pero queda demostrar)
class myDecoder_ver_3(nn.Module):
    """A LSTM decoder step with attention"""
    def __init__(self, embed_size: int, enc_hidden_size:int, dec_hidden_size: int, output_size:int, dropout_rate: float, endToken, seqEmb:nn.Embedding, device):
        super().__init__()
        self.device = device
        self.EOS = torch.tensor(endToken, device = self.device)
        self.maxDecodingTime = 100
        self.embed_size = embed_size

        self.decoder        = nn.LSTMCell(embed_size, dec_hidden_size, device=device)
        self.attention      = myAttention_ver_1(device=device)
        self.att_projection = nn.Linear(enc_hidden_size, dec_hidden_size, False, device=device)
        self.combOutput     = nn.Linear(2*enc_hidden_size, embed_size, False, device=device)
        self.dropout        = nn.Dropout(dropout_rate)      
        self.vocab_projection = nn.Linear(embed_size, output_size, False, device=device)
        self.seqEmb = seqEmb
        
    def forward(self,
                dec_initial_state:  Tuple[  TensorType['num_layers','batch_size','dec_hidden_size',float], 
                                            TensorType['num_layers','batch_size','dec_hidden_size',float]], 
                enc_hiddens:        TensorType["batch_size","src_len","enc_hidden_size", float], 
                enc_len:            TensorType["batch_size", int],
                tgt:                Tuple[  TensorType['batch_size','tgt_len',int],   
                                            TensorType['batch_size',int]]
                ) ->                TensorType['batch_size','tgt_len','tgt_vocab', int]:
        tgt = self.seqEmb(tgt[0]) if self.training else None
        EOS = self.seqEmb(self.EOS)
        dec_state = (dec_initial_state[0].squeeze(0),dec_initial_state[1].squeeze(0)) #delete num_layers dim

        decoding_time  = 0
        batch_size     = enc_len.shape[0]
        prevOut        = torch.zeros(batch_size, self.embed_size, device=self.device)
        tgt_len        = torch.zeros(batch_size, dtype=int)
        predictions    = []
        flags          = torch.tensor([False]*batch_size, device=self.device) #Flags when reached <end>

        values = enc_hiddens
        keys =  self.att_projection(enc_hiddens)
        self.attention.generate_sent_masks(enc_hiddens, enc_len)
        while decoding_time < self.maxDecodingTime:
            dec_input = prevOut  #TensorType['batch_size', 'embed_size']

            dec_state = self.decoder(dec_input, dec_state)
            c_t = self.attention(values, keys, dec_state[0])
    
            #Combining Input representation and Decoder state
            combined_output = torch.cat((dec_state[0], c_t), 1)    
            
            #Producing output from combined representation
            decOut = self.combOutput(combined_output) #From [dec_hidden; c_t] to embed_size
            decOut = torch.tanh(decOut)
            decOut = self.dropout(decOut)

            # symbols = self.greedyStrategy(out)

            predictions.append(decOut)
            prevOut = decOut

            #Forcing golden words
            if self.training: prevOut = tgt[:,decoding_time,:]  #TensorType['batch_size', 'embed_size']
            decoding_time += 1

            #Break when prediction has all <EOS>
            ###
            # 1. Estrategia de corte: Comparar con embedding <end>
            ixEOS = torch.all(torch.isclose(prevOut, EOS,rtol=1E-3),dim=1)
            # 2. Estrategia de corte: Proyectar en vocab tgt y obtener el maximo
            # vocabOut = self.vocab_projection(prevOut) #From embed_size to vocab_size
            # ixEOS = self.EOS == torch.argmax(vocabOut, dim=1)
            ###
            flags = torch.logical_or(flags, ixEOS)
            tgt_len[ixEOS] = decoding_time
            if torch.all(flags): break
        
        tgt_len[torch.logical_not(ixEOS)] = decoding_time #Saves final length

        embedPredictions = torch.stack(predictions,1) #(batch_size, tgt_len, dec_hidden_size)
        vocabPredictions = self.vocab_projection(embedPredictions) #From embed_size to Vocab_size
        #TODO: Esta estrategia no es consistente con la anterior!!!
        #   o las dos sacan una probabilidad
        #   o las dos sacan un embedding
        logP = F.log_softmax(vocabPredictions, dim=-1)
        return logP, tgt_len
