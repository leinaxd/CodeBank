from torchtyping import TensorType
import torch, torch.nn as nn


class heuristicDecoder(nn.Module):
    """
    Inference over source sequence with an heuristic
    
    Parameter: 
        - Heuristic module
        - Iterative function
    INPUT: 
    OUTPUT: Sequence
    """

    def __init__(self, model, heuristic, startToken, endToken, maxDecodingTime=200):
        super().__init__()
        self.heuristic = heuristic
        self.model = model
        self.startToken = startToken #SOS
        self.endToken = endToken #EOS
        self.maxDecodingTime = maxDecodingTime

    def forward(self, 
                srcRepr: TensorType["batch_size","seq_len","embedding_size"], 
                src_len: TensorType["batch_size"], 
                ) ->     TensorType["batch_size","tgt_len"]:
        batch_size = srcRepr.size(0)
        device = srcRepr.device

        predictions = torch.full([batch_size, 1], self.startToken, dtype=int, device=device)
        pred_len    = torch.ones(batch_size, dtype=int, device=device)
        flags       = torch.tensor([False]*batch_size, device=device) #Flags until reach <end>

        for decoding_time in range(1, self.maxDecodingTime):
            # print('decodingTime:', decoding_time)
            # print(predictions, pred_len)
            prediction = self.model(predictions, pred_len, srcRepr, src_len)
            # print(prediction)
            symbol = self.heuristic(prediction)
            predictions = torch.cat((predictions, symbol.unsqueeze(1)), 1) #concatenate through pred_len dim
            #Break when prediction has all <EOS>
            ixEOS = self.endToken == symbol #symbol.squeeze(-1)
            flags = torch.logical_or(flags, ixEOS)
            pred_len[torch.logical_not(flags)] = decoding_time+1 #save lengths while they didn't end + <SOS>
            if torch.all(flags): break

        return predictions[:,1:-1], pred_len-2 #Ignore <SOS> and <EOS>





        # prevToken = torch.zeros_like(srcRepr)
        # out = self.model(embTgt, srcRepr, alignMask, tgtMask)

        # embTgt  = self.tgt_embed(tgt)      
        # tgtMask = self.padMask(tgt_len, tgt_len)
        # # tgtMask &= self.causalMask(tgt)
        # alignMask = self.padMask(tgt_len, src_len)
        # out = self.predictDecoder(embTgt, srcRepr, alignMask, tgtMask)
        # out = self.outputLayer(out)
        # return out, tgt_len
        # self.heuristic()

