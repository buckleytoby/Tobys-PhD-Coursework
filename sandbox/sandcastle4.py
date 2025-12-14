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
1d temporal data. IRL we'd expect to save xt-1 = xt with each new observation. So adjacent elements are correlated.
Previously you'd create a Unet (or more recently a transformer), and just feed everything into it.

Hypothesis:
    learn to fit "attention" gaussians to the sequence, each gaussian outputs 
        learnable params: 1. [pos-mean, pos-std] --> this should fail with this dataset because the interesting subset of histories is always moving backwards
                            2. NN which outputs [pos-mean, pos-std] --> this should succeed with this dataset because then it can "track" the interesting neurons as data moves through the history
        outputs: [val-weighted-mean, val-weighted-std, pos-mean, pos-std]

    counter:
        attention does something similar, because it uses the same weights for each entry in a sequence IIRC

inputs:
    sequence of vals denoting a shape
outputs:
    1-hot shape classification

"""
    
def get_batch():
    h = 16 # history length
    b = BATCH_SIZE
    w = 1 # +- w around the center index
    
    imin0 = 1
    imax0 = 15

    imin1 = 1
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
        x[p1, i] = 1.1
        x[p1, imax] = 1.0


        # triangle: ____/^\_____
        i = np.random.randint(imin1, imax1, (nb_1,)) # np.random.randint(0, h)
        imin = i - w
        imax = i + w
        imin[imin<0] = 0
        imax[imax > h] = h


        x[p2, imin] = 1.0
        x[p2, i] = 2.0
        x[p2, imax] = 1.0
        
        # x /= x.sum(axis=1, keepdims=True)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        yield x, y

class SoftSlice1D(nn.Module):
    """
    set window size. NN to output window center

    basically a soft-array slice, similar to how attention is like a soft-dictionary lookup
    """
    def __init__(self, 
                 in_w,
                 kernel_size,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_w = in_w
        self.kernel_size = kernel_size

        self.out_w = 1 + 2 * self.kernel_size
        
        n1 = 128
        self.window_nn = nn.Sequential(
            nn.Linear(in_w, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, 1),
            nn.Tanh(),
            # spatialsoftargmax()
        )
        
        
        xpos = np.linspace(-1., 1., self.in_w)
        xpos2 = torch.from_numpy(xpos).float().to(DEVICE)

        self.xpos = torch.unsqueeze(xpos2, dim=0)
        

        self.register_full_backward_hook(self.module_backward_hook)
        
    def forward(self, inputs):
        b = inputs.shape[0]

        # cat the inputs and the xpos's
        # x = torch.cat((inputs, self.xpos), dim=-1)
        
        # range [-1, 1]
        window_center = self.window_nn(inputs)
        window_center.retain_grad()
        self.window_center = window_center
        
        # idx = (window_center.detach() + self.in_w / 2.0).to(torch.int).squeeze()

        # differentiable
        softidx = window_center * self.in_w / 2.0 + self.in_w / 2.0

        # distance function - abs
        d = lambda x, y: torch.abs(x - y)

        floors = torch.floor(softidx).to(torch.int)
        ceils = torch.ceil(softidx).to(torch.int)

        idxs0 = torch.range(-self.kernel_size, self.kernel_size).to(torch.int)

        idxs1 = idxs0 + floors
        idxs2 = idxs0 + ceils

        w1s = 1.0 - d(softidx, floors)
        w2s = 1.0 - d(softidx, ceils)

        # construct the A matrix, shape [b, out_w, in_w]
        A = torch.zeros((b, self.out_w, self.in_w))
        B = torch.zeros((b, self.out_w, self.in_w))

        y = torch.empty((b, self.out_w))

        for i in range(b):
            y[i] = inputs[i][idxs1[i]] * w1s[i] + inputs[i][idxs2[i]] * w2s[i]

        # y = A @ inputs
        y.retain_grad()
        self.y = y
                
        return y
        


    def module_backward_hook(self, module, grad_input, grad_output):
        # print("test")
        pass

    # return grad_input, grad_output

class SoftSlice2D(nn.Module):
    """
    set window size. NN to output window center

    basically a soft-array slice, similar to how attention is like a soft-dictionary lookup
    """
    def __init__(self, 
                 in_w,
                 kernel_size,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_w = in_w
        self.kernel_size = kernel_size

        self.out_w = 1 + 2 * self.kernel_size
        
        n1 = 128
        self.window_nn = nn.Sequential(
            nn.Linear(in_w, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, 1),
            nn.Tanh(),
            # spatialsoftargmax()
        )
        
        
        xpos = np.linspace(-1., 1., self.in_w)
        xpos2 = torch.from_numpy(xpos).float().to(DEVICE)

        self.xpos = torch.unsqueeze(xpos2, dim=0)
        

        self.register_full_backward_hook(self.module_backward_hook)
        
    def forward(self, inputs):
        b = inputs.shape[0]

        # cat the inputs and the xpos's
        # x = torch.cat((inputs, self.xpos), dim=-1)
        
        # range [-1, 1]
        window_center = self.window_nn(inputs)
        window_center.retain_grad()
        self.window_center = window_center
        
        # idx = (window_center.detach() + self.in_w / 2.0).to(torch.int).squeeze()

        # differentiable
        softidx = window_center * self.in_w / 2.0 + self.in_w / 2.0

        # distance function - abs
        d = lambda x, y: torch.abs(x - y)

        floors = torch.floor(softidx).to(torch.int)
        ceils = torch.ceil(softidx).to(torch.int)

        idxs0 = torch.range(-self.kernel_size, self.kernel_size).to(torch.int)

        idxs1 = idxs0 + floors
        idxs2 = idxs0 + ceils

        w1s = 1.0 - d(softidx, floors)
        w2s = 1.0 - d(softidx, ceils)

        # construct the A matrix, shape [b, out_w, in_w]
        A = torch.zeros((b, self.out_w, self.in_w))
        B = torch.zeros((b, self.out_w, self.in_w))

        y = torch.empty((b, self.out_w))

        for i in range(b):
            y[i] = inputs[i][idxs1[i]] * w1s[i] + inputs[i][idxs2[i]] * w2s[i]

        # y = A @ inputs
        y.retain_grad()
        self.y = y
                
        return y

def get_num_correct(preds, labels):
    # preds.argmax(dim=1) gets the index of the max log-probability/logit (the predicted class)
    # .eq(labels) checks where the predictions equal the actual labels, returning a boolean tensor
    # .sum() calculates the total number of correct predictions in the batch
    nb = preds.argmax(dim=1).eq(labels).sum().item()
    sr = nb / len(labels)
    return sr

def main():
    nb_inputs = 16
    kernel_size = 2
    

    SoftSlice1D_output = 1 + 2 * kernel_size
    
    nb_hidden = 64

    nb_outputs = 2
    
    # make a simple NN
    model = nn.Sequential(
        # 1. baseline
        # nn.Linear(nb_inputs, SoftSlice1D_output),
        # nn.LeakyReLU(),

        # 3. 
        SoftSlice1D(in_w=nb_inputs, kernel_size=kernel_size),

        nn.Linear(SoftSlice1D_output, nb_hidden),
        nn.LeakyReLU(),

        # final layer to get outputs
        nn.Linear(nb_hidden, nb_outputs),
    )
    model = model.to(DEVICE)

    # optimizer - weight_decay should be small ~1e-3
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-9)

    # classification loss
    loss_fn = torch.nn.CrossEntropyLoss() # applies softmax to its input


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

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

                opt.zero_grad()


                outputs = model(inputs)

                l = loss_fn(outputs, labels)
                l.retain_grad()
                l.backward(retain_graph=True)

                opt.step()

                # SR
                sr = get_num_correct(outputs, labels)

                # avg metrics
                metrics['avg_l'] = 0.99 * metrics['avg_l'] + 0.01 * float(l.detach())
                metrics['avg_sr'] = 0.99 * metrics['avg_sr'] + 0.01 * float(sr)


                t.set_postfix(metrics)
                t.update()

        # end of epoch
        # print(pp.probabilities)

        model.eval()

        # validation

        # reset to training mode
        model.train()

        if False:
            # p = pds.productivity.clone().detach()
            # p = pds.get_productive_inputs()
            inds = roaming_gaus.gaus

            p = torch.zeros(28*28)
            p[inds] = 1.0

            p2 = torch.unflatten(p, 0, (28, 28))

            p2 = p.reshape([28, 28])

            # [0, 1]
            p2 /= p2.max()

            if PLOT:
                p3 = p2.cpu()
                plt.clf()
                plt.imshow(p3, vmin=0, vmax=1)
                plt.show()
                plt.colorbar()
                fig.canvas.draw()
                fig.canvas.flush_events()


        


    pass
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

main()