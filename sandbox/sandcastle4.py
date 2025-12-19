

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


turns out this is already solved. 
https://docs.pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html


"""
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

PLOT = False

if PLOT:
    plt.ion()
    # fig = plt.figure()


BATCH_SIZE = 64
DEVICE = 'cpu' # must be cpu for backward hook to work

transform = transforms.Compose(
[transforms.ToTensor(),
#  transforms.Lambda(lambda x: torch.flatten(x)),
 ])

# 60k datapts
mnist = MNIST(".", download=True, transform=transform, train=True)

data_loader = DataLoader(mnist,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                    )
    
def get_batch_1():
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



def get_mnist():
    global data_loader
    
    while True:
        for idx, (x, y) in enumerate(data_loader):
            x = x.squeeze(dim=1)
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
                 nb_kernels,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_w = in_w
        self.kernel_size = kernel_size
        self.nb_kernels = nb_kernels

        self.out_w = 1 + 2 * self.kernel_size

        ndim = 2
        
        n1 = 128

        # in the future this could be any arbitrary architecture (cnn or vit). For the PoC though just use a dense network
        self.window_nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_w * in_w, n1),

            nn.LeakyReLU(),
            nn.Linear(n1, n1),

            nn.LeakyReLU(),
            nn.Linear(n1, n1),

            nn.LeakyReLU(),
            nn.Linear(n1, n1),

            nn.LeakyReLU(),
            nn.Linear(n1, ndim * self.nb_kernels),

            nn.Tanh(),
            # spatialsoftargmax()
            nn.Unflatten(dim=1, unflattened_size=(self.nb_kernels, ndim)),
        )
        
        
        xpos = np.linspace(-1., 1., self.in_w)
        xpos2 = torch.from_numpy(xpos).float().to(DEVICE)

        xpos3 = torch.unsqueeze(xpos2, dim=0)
        
    def forward_OBS(self, inputs):
        b = inputs.shape[0]
        y = torch.empty((b, self.nb_kernels, self.out_w, self.out_w))

        # cat the inputs and the xpos's
        # x = torch.cat((inputs, self.xpos), dim=-1)
        
        # range [-1, 1]
        window_centers = self.window_nn(inputs)
            
        for j in range(self.nb_kernels):
            window_center = window_centers[:, j, :]

            # square output
            softidx = window_center * self.in_w / 2.0 + self.in_w / 2.0

            floors = torch.floor(softidx).to(torch.int)
            ceils = torch.ceil(softidx).to(torch.int)

            # must use meshgrid for advanced 2d indexing
            ir = torch.range(-self.kernel_size, self.kernel_size).to(torch.int)
            ix0, iy0 = torch.meshgrid([ir, ir])
            ix0 = ix0.unsqueeze(dim=0)
            iy0 = iy0.unsqueeze(dim=0)
            # ix0 = torch.range(-self.kernel_size, self.kernel_size).to(torch.int).unsqueeze(dim=0)
            # iy0 = torch.range(-self.kernel_size, self.kernel_size).to(torch.int).unsqueeze(dim=0)

            ix1 = ix0 + floors[:, 0:1].unsqueeze(dim=-1)
            iy1 = iy0 + floors[:, 1:2].unsqueeze(dim=-1)

            ix2 = ix0 + ceils[:, 0:1].unsqueeze(dim=-1)
            iy2 = iy0 + ceils[:, 1:2].unsqueeze(dim=-1)

            ix1 = torch.clip(ix1, 0, self.in_w-1)
            iy1 = torch.clip(iy1, 0, self.in_w-1)
            ix2 = torch.clip(ix2, 0, self.in_w-1)
            iy2 = torch.clip(iy2, 0, self.in_w-1)

            # weight function
            def w(floors, softidx):
                l1 = softidx[:, 0] - floors[:, 0]
                l2 = 1.0 - l1

                w1 = softidx[:, 1] - floors[:, 1]
                w2 = 1.0 - w1

                a1 = l1 * w1 # x1, y1
                a2 = l1 * w2 # x1, y2
                a3 = l2 * w1 # x2, y1
                a4 = l2 * w2 # x2, y2 or 1.0 - a1 - a2 - a3
                return a1, a2, a3, a4

            w1s, w2s, w3s, w4s = w(floors, softidx)

            for i in range(b):
                ix1i = ix1[i]
                ix2i = ix2[i]
                iy1i = iy1[i]
                iy2i = iy2[i]

                # oobx1 = ix1 < 0 or ix1 >= self.in_w
                
                ib = inputs[i]
                o1 = ib[ix1[i], iy1[i]]
                o2 = ib[ix1[i], iy2[i]]
                o3 = ib[ix2[i], iy1[i]]
                o4 = ib[ix2[i], iy2[i]]

                y[i, j] = w1s[i] * o1 + w2s[i] * o2 + w3s[i] * o3 + w4s[i] * o4

        # y = A @ inputs
        y.retain_grad()
        self.y = y
                
        return y
        
    def forward(self, inputs):
        b = inputs.shape[0]
        y = torch.empty((b, self.nb_kernels, self.out_w, self.out_w))

        # cat the inputs and the xpos's
        # x = torch.cat((inputs, self.xpos), dim=-1)
        
        # range [-1, 1]
        window_centers = self.window_nn(inputs)

        softidx = window_centers * self.in_w / 2.0 + self.in_w / 2.0

        floors = torch.floor(softidx).to(torch.int)
        ceils = torch.ceil(softidx).to(torch.int)

        # must use meshgrid for advanced 2d indexing
        ir = torch.range(-self.kernel_size, self.kernel_size).to(torch.int)
        ix0, iy0 = torch.meshgrid([ir, ir])
        ix0 = ix0.unsqueeze(dim=0)
        iy0 = iy0.unsqueeze(dim=0)

        ix1 = ix0 + floors[..., 0:1].unsqueeze(dim=-1)
        iy1 = iy0 + floors[..., 1:2].unsqueeze(dim=-1)

        ix2 = ix0 + ceils[..., 0:1].unsqueeze(dim=-1)
        iy2 = iy0 + ceils[..., 1:2].unsqueeze(dim=-1)

        ix1 = torch.clip(ix1, 0, self.in_w-1)
        iy1 = torch.clip(iy1, 0, self.in_w-1)
        ix2 = torch.clip(ix2, 0, self.in_w-1)
        iy2 = torch.clip(iy2, 0, self.in_w-1)

        # weight function
        def w(floors, softidx):
            l1w1 = (softidx - floors)
            l2w2 = (1.0 - l1w1)

            l1w1l2w2 = torch.cat([l1w1, l2w2], dim=2)
            reorder = [0, 2, 1, 3]

            l1l2w1w2 = l1w1l2w2[..., reorder]

            l1l2 = l1l2w1w2[..., 0:2].unsqueeze(dim=-1)
            w1w2 = l1l2w1w2[..., 2:4].unsqueeze(dim=-1).permute([0, 1, 3, 2])

            # want ...x2x1 @ ...x1x2 -> ......x2x2
            a14 = l1l2 @ w1w2

            return a14
        
        w14 = w(floors, softidx)

        i2 = inputs.unsqueeze(dim=1)
        i3 = i2.tile((1, self.nb_kernels, 1, 1))
        y = w14 @ inputs
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
    nb_inputs = 28
    kernel_size = 2
    nb_kernels = 16
    

    SoftSlice1D_output = 1 + 2 * kernel_size
    
    nb_hidden = 64

    nb_outputs = 10
    
    # make a simple NN
    model = nn.Sequential(
        # 1. baseline
        # nn.Flatten(),
        # nn.Linear(nb_inputs**2, SoftSlice1D_output**2),
        # nn.LeakyReLU(),

        # 2. 
        SoftSlice2D(in_w=nb_inputs, kernel_size=kernel_size, nb_kernels=nb_kernels),

        nn.Flatten(),
        nn.Linear(SoftSlice1D_output**2 * nb_kernels, nb_hidden),
        nn.LeakyReLU(),

        nn.Linear(nb_hidden, nb_hidden),
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

    metrics = {"avg_l": 10.0, "avg_sr": 0.0}

    nb_epochs = 10000
    nb_batches = 1000
    for _ in range(nb_epochs): # tqdm.tqdm(range(nb_epochs)):
        # reset


        with tqdm.tqdm(postfix=metrics) as t:
            for idx, batch in enumerate(get_mnist()):
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
                metrics['avg_l'] = 0.95 * metrics['avg_l'] + 0.05 * float(l.detach())
                metrics['avg_sr'] = 0.95 * metrics['avg_sr'] + 0.05 * float(sr)


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