import torch
import torch.nn as nn

class seq2seq(nn.Module):
    """Transforms <n> inputs sequences into an <m> output sequences"""
    #Generalizar a n secuencias de entrada y m de salida
    #   Ej. Series temporales (electroencefalograma)
    def __init__(self, n: int, m:int):
        super().__init__()


class my2seq1seq(nn.Module):
    """Transforms 2 input sequences into an output sequence"""
    def __init__(self, vocab, embed_size, hidden_size, device):
        super().__init__()
        self.vocab = vocab
        input_size = len(vocab.getVocab())
        output_size = input_size
        dropout_rate = 0
        self.seqEmb = nn.Embedding(input_size, embed_size, *vocab.w2i('<pad>'))
        self.ctxEncoder = myEncoder_ver_1(embed_size,hidden_size)
        self.qEncoder   = myEncoder_ver_1(embed_size,hidden_size)
        self.aDecoder   = myDecoder_ver_1(embed_size,hidden_size,output_size,dropout_rate, *vocab.w2i('<pad>'), *vocab.w2i('<end>'), device=device)
    def forward(self, context: Tuple[TensorType['batch_size', 'context_len',int],  TensorType['batch_size',1]], 
                      question:Tuple[TensorType['batch_size', 'question_len',int], TensorType['batch_size',1]], 
                      answer:  Tuple[TensorType['batch_size', 'answer_len',int],   TensorType['batch_size',1]]
                      )->      TensorType['batch_size', 'answer_len','vocab_len',int]:

        context  = (self.seqEmb(context[0]), context[1]) #Input
        question = (self.seqEmb(question[0]), question[1]) #Input
        answer   = (self.seqEmb(answer[0]), answer[1]) #Input

        enc_ctx, enc_final_state = self.ctxEncoder(*context)
        enc_q, enc_final_state   = self.qEncoder(*question, enc_final_state)
        
        enc_len     = torch.cat((context[1], question[1]), 0)
        enc_hiddens = torch.cat((enc_ctx, enc_q), 1) #Appends hiddens

        print(enc_len.shape)
        print(enc_hiddens.shape)
        raise NotImplementedError
        dec_hiddens, dec_final_state = self.aDecoder(enc_final_state, enc_hiddens, enc_len, answer)
        return 0
        
    def padBatch(self, batchedSequence, device):
        """pads a variable length sequence into a fixed tensor"""
        seq_len = [len(sequence) for sequence in batchedSequence] 
        max_length = max(seq_len)
        seq = [sequence + self.vocab.w2i('<pad>')*(max_length-len(sequence)) for sequence in batchedSequence]
        seq = torch.tensor(seq, dtype=torch.long, device=device)# Tensor: (b, seq_len)
        seq_len = torch.tensor(seq_len, device=device)
        return seq, seq_len

    def loss(self, trainBatch, device):
        c, q, a = trainBatch

        c, c_len = self.padBatch(c, device) #c = batch x context_len
        q, q_len = self.padBatch(q, device) #q = batch x question_len 
        a, a_len = self.padBatch(a, device) #a = batch x answer_len

        prediction = self.forward((c,c_len),(q,q_len),(a,a_len))
        raise NotImplementedError
        loss = nn.CrossEntropyLoss(logits, y)
        return loss