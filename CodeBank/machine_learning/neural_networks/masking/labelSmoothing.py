
import torch, torch.nn as nn
from torchtyping import TensorType
class labelSmoothing(nn.Module):
    """Implement label smoothing. with Kullback Leibler loss function

    We create a distribution that has confidence on the correct word,
    and the rest of mass is smoothed throughout the vocabulary.

    Label Smoothing penalizes the model if it gets very confident about a given choise
    Parameters
        <confidence>: Amount of probability to difuse

    Input
        tgt_ix: The true answers indexes

    Output
        A smoothed true answer distribution

    Method
        Higher value = <confident>
        Lower value = (1-<smoothing>)/<seq_len>

    Applications
        Kullback Leibler Divergence, Usefull as it doesn't work for zero probability

    Paper? 
        https://arxiv.org/abs/1512.00567
    """
    def __init__(self, confidence:float=0.8):
        super().__init__()
        self.confidence = confidence
        self.smoothing = 1.0-confidence
    def forward(self, prediction: TensorType['batch_size','seq_len','output_size'],
                             ix : TensorType['batch_size','seq_len'],
                          )  ->   TensorType['batch_size','seq_len','output_size']:
        # seq_len = prediction.size(1)
        size=prediction.shape
        device = prediction.device
        if isinstance(ix, list): ix = torch.tensor(ix, device=device)
        # print(ix)
        # ix = torch.flatten(ix)
        output_size = size[-1]
        zeroCount = output_size-1 #distribute uniform at zeros
        assert output_size > 1, f"you cannot smooth an output_prob_dim <= 1"
        assert self.smoothing/zeroCount<self.confidence, f"High values should be greather than lower values"
        # Define lower value: self.smoothing/(seq_len-2)
        # Define high value: self.confidence
        # true_dist.fill_(self.smoothing / (seq_len - 2))

        #create a smoothed Window
        smoothed_dist = torch.full(size, self.smoothing/zeroCount, device=device)
        smoothed_dist.scatter_(-1, ix.unsqueeze(-1), self.confidence)
        # true_dist = torch.full_like(prediction, self.smoothing/(seq_len-2))
        # true_dist.scatter_(1, tgt_ix.unsqueeze(1), self.confidence)
        return smoothed_dist

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    test = 1
    if test == 1:
        confidence = 1
        distribution = labelSmoothing(confidence)
        predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                                [0, 0.2, 0.7, 0.1, 0], 
                                [0, 0.2, 0.7, 0.1, 0]])
        true = [4,1,0]
        # true = [[4],[1],[0]]
        lPred = predict.log()
        lTrue = torch.LongTensor(true)
        smoothed_dist = distribution(lPred, lTrue)
        print(predict)
        print(true)
        print(smoothed_dist)
        true = [4,10,0]
        lTrue = torch.LongTensor(true)
        print(f'expected error at index 1:\n {true}')
        smoothed_dist = distribution(lPred, lTrue)
    if test == 2:
        confidence = 1
        confidence = 0.8
        confidence = 0.3
        # Example of label smoothing.
        distribution = labelSmoothing(confidence)
        predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                                [0, 0.2, 0.7, 0.1, 0], 
                                [0, 0.2, 0.7, 0.1, 0]])
        lPred = predict.log()
        lTrue = torch.LongTensor([2, 1, 0])
        smoothed_dist = distribution(lPred, lTrue)
        fig = plt.figure()
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)
        plt.tight_layout()
        # Show the target distributions expected by the system.
        tgt = torch.zeros_like(predict)
        for i in range(3): tgt[i, lTrue[i]] = 1
        ax1.set_title('true distribution')
        line = ax1.imshow(tgt, vmin=0, vmax=1)
        ax1.set_ylabel('samples')
        ax1.set_xlabel('category')
        ax2.set_title('predicted distribution')
        ax2.imshow(predict, vmin=0, vmax=1)
        ax2.set_xlabel('category')
        ax3.set_title('smoothed distribution')
        ax3.imshow(smoothed_dist, vmin=0, vmax=1)
        ax3.set_xlabel('category')

        fig.colorbar(line, ax=[ax1,ax2,ax3], location='bottom')
        plt.show()
    if test == 3:
        def predict(x):
            ones = torch.ones_like(x)
            predict = torch.cat([x, ones, ones, ones], dim=1)
            den = torch.sum(predict, 1, True)#prob sums to one
            predict = predict / den
            return predict

        x = torch.arange(1, 100).unsqueeze(1)
        y = predict(x).log()
        tgt_true = torch.full([99],0)

        print('Explanation, Higher values of x, means higher probability at dim 0, so the lower the loss function')
        print('True value, category 0')
        print('Predicted value, [x, 1, 1, 1]/sum(x,3')
        confidence = 0.9
        true_dist = labelSmoothing(1.0)
        smoothed_dist = labelSmoothing(confidence)
        
        crit = torch.nn.KLDivLoss(reduction='none')
        
        lTrue= true_dist([99,4], tgt_true)
        loss_true = crit(y, lTrue).sum(1)

        lSmoothed = smoothed_dist([99,4], tgt_true)
        loss_smoothed = crit(y, lSmoothed).sum(1)
        print(loss_smoothed.shape)
        plt.plot(x, loss_true)
        plt.plot(x, loss_smoothed)
        plt.legend(['loss true category', 'loss smoothed categories'])
        plt.show()
    if test == 4:
        confidence = 0.9
        # Example of label smoothing.
        distribution = labelSmoothing(confidence)
        predict = torch.tensor([[[0.5,   0, 0.5, 0, 0],
                                 [0.6,   0, 0.4, 0, 0], 
                                 [0.7,   0, 0.3, 0, 0]],
                                [[  0, 0.5, 0.5, 0, 0],
                                 [  0, 0.6, 0.4, 0, 0], 
                                 [  0, 0.7, 0.3, 0, 0]],
                                ])
        lPred = predict.log()
        lTrue = torch.LongTensor([[2,0,0],[2,1,1]])
        print(predict.shape)
        print(lTrue.shape)
        smoothed_dist = distribution(lPred, lTrue)
        #Show the target distributions expected by the system.
        tgt = torch.zeros_like(predict)
        for i in range(2): 
            for s in range(3):
                tgt[i, s, lTrue[i,s]] = 1
        #Plot
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle(f"sample 1")
        axs = subfigs[0].subplots(nrows=1, ncols = 3)
        axs[0].set_title('true distribution')
        line = axs[0].imshow(tgt[0], vmin=0, vmax=1)
        axs[0].set_ylabel('seq_pos')
        axs[1].set_title('predicted distribution')
        axs[1].imshow(predict[0], vmin=0, vmax=1)
        axs[2].set_title('smoothed distribution')
        axs[2].imshow(smoothed_dist[0], vmin=0, vmax=1)

        subfigs[1].suptitle(f"sample 2")
        axs = subfigs[1].subplots(nrows=1, ncols = 3)
        axs[0].imshow(tgt[1], vmin=0, vmax=1)
        axs[0].set_ylabel('seq_pos')
        axs[0].set_xlabel('category')
        axs[1].imshow(predict[1], vmin=0, vmax=1)
        axs[1].set_xlabel('category')
        axs[2].imshow(smoothed_dist[1], vmin=0, vmax=1)
        axs[2].set_xlabel('category')

        fig.colorbar(line, ax=axs, location='bottom')
        # fig.suptitle('Sample 0 and Sample 1')
        # plt.tight_layout()
        plt.show()
        