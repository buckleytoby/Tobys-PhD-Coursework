import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal

PLOT = True

if PLOT:
    plt.ion()
    # fig = plt.figure()


BATCH_SIZE = 64
DEVICE = 'cpu' # must be cpu for backward hook to work

"""
Sandcastle 5 - inspiration: ffmpeg will have key-frames periodically which take much longer to compress, and then inbetween those key-frames they'll have delta-frames so they only have to process a small amount of data, resulting in much quicker compression of non key-frame frames.

Design a neural net that exploits temporal relationship of inputs
key-frame ~~ absolute obs and state inputs
delta-frames ~~ key-frame - current-frame

flavors:
RNN
ffmpeg

this is pretty similar to an RNN, and LSTM's
"""

    
def get_batch():
    h = 16 # history length
    b = BATCH_SIZE
    w = 1 # +- w around the center index
    
    imin0 = 1
    imax0 = 2

    imin1 = 14
    imax1 = 15

    while True:
        x = np.zeros((b, h))
        y = np.zeros((b,))
        
        p = np.random.random((b,))
        
        p1 = p > 0.5
        p2 = ~p1

        nb_0 = p1.sum()
        nb_1 = p2.sum()
        
        y[p1] = 0.0
        y[p2] = 1.0
        
        # rectangle: ____----_____
        i = np.random.randint(imin0, imax0, size = (nb_0,)) #  np.random.randint(0, h)
        imin = i - w
        imax = i + w
        imin[imin<0] = 0
        imax[imax > h] = h

        x[p1, imin] = 1.0
        x[p1, i] = 1.0
        x[p1, imax] = 1.0


        # triangle: ____/^\_____
        i = np.random.randint(imin1, imax1, (nb_1,)) # np.random.randint(0, h)
        imin = i - w
        imax = i + w
        imin[imin<0] = 0
        imax[imax > h] = h


        x[p2, imin] = 1.0
        x[p2, i] = 1.0
        x[p2, imax] = 1.0
        
        # x /= x.sum(axis=1, keepdims=True)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        yield x, y

class KeyFrameNN(nn.Module):
    """
    just a normal NN
    """
    def __init__(self, 
                 nb_inputs,
                 nb_hidden,
                 nb_outputs,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(nb_inputs, nb_hidden),
            nn.LeakyReLU(),
            nn.Linear(nb_hidden, nb_hidden),
            nn.LeakyReLU(),
            nn.Linear(nb_hidden, nb_outputs),
        )

        def get_nb_parameterized_layers(self):
            """
            nb parameterized layers 
            """
            return 2

        def forward(self, inputs):
            return self.model(inputs)
        
class DeltaNN(nn.Module):
    """
    given inputs xk and delta-xi, predict yi
    """
    def __init__(self, 
                 nb_keyframe_inputs,
                 nb_hidden,
                 nb_outputs,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(nb_keyframe_inputs + nb_outputs, nb_hidden),
            nn.LeakyReLU(),
            nn.Linear(nb_hidden, nb_outputs),
        )

        def forward(self, keyframe_inputs, delta_inputs):
            x = torch.cat([keyframe_inputs, delta_inputs])
            return self.model(x)

def main():
    nb_inputs = 16
    kernel_size = 2
    

    nnwindow_output = 1 + 2 * kernel_size
    
    nb_hidden = 64

    nb_outputs = 2
    
    # make a simple NN
    keyframe_model = KeyFrameNN(nb_inputs, nb_hidden, nb_outputs)

    #
    delta_model = DeltaNN(nb_inputs, nb_hidden, nb_outputs)

    # optimizer - weight_decay should be small ~1e-3
    keyframe_opt = torch.optim.SGD(keyframe_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-9)
    delta_opt = torch.optim.SGD(keyframe_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-9)


    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    metrics = {"avg_l": 0.0, "avg_sr": 0.0}

    nb_epochs = 10000
    nb_batches = 1000
    for _ in range(nb_epochs): # tqdm.tqdm(range(nb_epochs)):
        # reset


        with tqdm.tqdm(postfix=metrics) as t:
            for idx, batch in enumerate(get_batch()):
                inputs, labels = batch
                
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                if idx > nb_batches:
                    break

                # train keyframe model
                if True:
                    keyframe_opt.zero_grad()


                    outputs = keyframe_model(inputs)

                    l = loss_fn(outputs, labels)
                    l.backward(retain_graph=True)

                    keyframe_opt.step()

                # train delta model
                if True:
                    outputsk = outputs.clone().detach()

                    delta_outputsi = delta_model(inputsk, delta_inputs, outputsk)

                    outputsi = delta_outputsi + outputsk

                    # g.t. outputsi from the keyframe_model
                    with torch.no_grad():
                        outputsi_gt = keyframe_model(inputsi)

                    l = loss_fn(outputsi, outputsi_gt)
                    l.backward()

                    delta_opt.step()


                # avg metrics
                metrics['avg_l'] = 0.99 * metrics['avg_l'] + 0.01 * float(l.detach())
                metrics['avg_sr'] = 0.99 * metrics['avg_sr'] + 0.01 * float(sr)


                t.set_postfix(metrics)
                t.update()

        # end of epoch

        


    pass

main()