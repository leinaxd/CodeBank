from torchtyping import TensorType
import torch, torch.nn as nn
import math


class positionalEmbedding(nn.Module):
    """
    Applies an additive transformation to a <sequence> of embeddings <x>
        return <x1 x2 .. xm> + <pos_enc(1) pos_enc(2) .. pos_enc(m)>

    The transformation is given by
        pos_enc(pos, dim):
            sin(pos·1/<max_len>^{2*dim/<embed_size>}) if pos is even
            cos(pos·1/<max_len>^{2*dim/<embed_size>}) if pos is odd

    Tutorials.
        The annotated transformer
        - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, embed_size, max_len=10000):
        super().__init__()
#       Sirve el dropout en los embeddings?
#         self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(max_len, embed_size)
        pos_ix = torch.arange(0, max_len).unsqueeze(1)
        dim_ix = torch.arange(0, embed_size, 2)
        w = torch.exp(-math.log(max_len)*dim_ix/embed_size )
        #   1/10000 ^ dim_ix/embed_size
        #   exp(log(1/10000 ^ dim_ix/embed_size))
        #   exp(log(1/10000) * dim_ix/embed_size)
        #   exp(-log(10000)  * dim_ix/embed_size)
        pos_enc[:, 0::2] = torch.sin(w * pos_ix) #even
        pos_enc[:, 1::2] = torch.cos(w * pos_ix) #odd
        pos_enc.unsqueeze_(0) #batch dim
        self.register_buffer('pos_enc', pos_enc) #this way you can move to any device from module

    def forward(self, x:TensorType['batch_size', 'seq_len','embed_size']) -> TensorType['batch_size','seq_len','embed_size']:
        return x+self.pos_enc[:, :x.size(1)]
#         return self.dropout(x)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    test = 3
    embed_size = 8
    seq_len = 50
    embedding_1 = nn.Embedding(100, embed_size)
    embedding_2 = positionalEmbedding(embed_size, 10000)
    if test == 1:
        data = torch.tensor([[10,20,30,40],[2,4,6,8]]) #batch_size = 2, seq_len = 4
        out = embedding_1(data) #batch_size = 2, seq_len = 4, embedding_size = 10
        out = embedding_2(out) #batch_size = 2, seq_len = 4, embedding_size = 10
        print('data:', data.shape)
        print('out_shape:', out.shape)
    if test == 2:
        out = torch.zeros(1,seq_len, embed_size)
        out = embedding_2(out) #batch_size = 2, seq_len = 4, embedding_size = 10
        plt.figure(figsize=(15, 5))
        axEven = plt.subplot(2,1,1)
        axOdd = plt.subplot(2,1,2)
        x = (torch.arange(0, seq_len)).tolist()
        # x = (torch.arange(0, seq_len)/torch.pi).tolist()
        dim_range = range(0, embed_size, 2)
        yEven = out[0, :, dim_range].tolist()
        axEven.plot(x,yEven)
        axEven.set_ylabel('sin(w pos)')
        axEven.legend([f"dim {p}" for p in dim_range], loc='lower left')
        dim_range = range(1, embed_size, 2)
        yOdd  = out[0, :, dim_range].tolist()
        axOdd.plot(x,yOdd)
        axOdd.legend([f"dim {p}" for p in dim_range], loc='lower left')
        axOdd.set_ylabel('cos(w pos)')
        axOdd.set_xlabel('seq_len [pi]')
        plt.show()
    if test == 3:
        print('verificar que cada embedding es diferente')
        out = torch.zeros(1,seq_len, embed_size)
        out = embedding_2(out) #batch_size = 2, seq_len = 4, embedding_size = 10
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot()
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05,right=0.95,top=0.99,bottom=0.15)
        line = ax.imshow(out[0].T, cmap='coolwarm')
        ax.invert_yaxis()
        fig.colorbar(line, location='bottom', orientation='horizontal')
        ax.set_xlabel('seq_len')
        ax.set_ylabel('d_emb')
        ax.set_title('positional encoding')
        # plt.tick_params(labelbottom=False, labeltop=True)
        plt.show()

#Input a sequence
#output an encoded sequence


#-> Con sentence pieces podemos controlar el tamaño
#-> del embedding

#-> Hasta tokens de longitud 10000
#-> Posiciones cercanas, tienen representaciones cercanas. Posiciones lejanas, representaciones lejanas


#Puede el positional encoding ser exactamente una senoidal entera? así sabe que

#LA IDEA ES AÑADIR UN VECTOR DE POSICION
# PERO NO DESTRUYA LA INFORMACION DE CERCANIA


#-> Self-attention sin softmax que es ineficiente

#-> Q, K en realidad no tienen por qué ser matrices diferentes, pueden ser una simple

#-> Self attention, agregar diferente

#-> K,V = diccionario. Q = consulta, pedir un conjunto de K
#-> La matriz QK^t utiliza toda la secuencia.
#   Aplicar la máscara es no ver los tokens futuros 
#   

#Cada símbolo tiene su representación? o cada secuencia tiene su representación self-attention?
