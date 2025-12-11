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
DEVICE = 'cpu'

    
def get_batch():
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
    h = 16 # history length
    b = 1
    w = 1 # +- w around the center index

    while True:
        p = np.random.rand()

        if p > 0.5:
            # rectangle: ____----_____
            i = np.random.randint(0, h)

            imin = max(0, i-w)
            imax = min(h, i+w+1)

            x = np.zeros((b, h))
            x[:, imin:imax] = 1.0

            x /= x.sum()

            y = [0]

        else:
            # triangle: ____/^\_____
            i = np.random.randint(0, h)
            imin = max(0, i-w)
            imax = min(h, i+w+1)

            x = np.zeros((b, h))
            x[:, imin:imax] = 1.0
            x[:, i] = 2.0

            x /= x.sum()

            y = [1]

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        yield x, y




class RoamingGaussianDownsample1DLearnableStats(nn.Module):
    """
    each output pixel has an associated roaming gaussian with a learnable mean and variance
    """
    def __init__(self, 
                 in_w,
                 nb_gaussians,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_w = in_w
        self.nb_gaussians = nb_gaussians

        xpos = np.linspace(-1., 1., self.in_w)
        xpos2 = torch.from_numpy(xpos).float()

        self.xpos = torch.tile(xpos2, (self.nb_gaussians, 1))

        self.mean = nn.Parameter(
            torch.rand((self.nb_gaussians, 1)) * 2.0 - 1.0
            )
        self.std = nn.Parameter(
            torch.rand((self.nb_gaussians, 1)) + 0.5
            )
        
    def log_prob(self, value):
        # Helper to calculate log probability of a value under the current distribution
        distribution = Normal(self.mean, self.std)
        return distribution.log_prob(value)

    def forward(self, inputs):
        # get the x positions
        xpos = self.xpos

        # Calculate log probability for each point
        log_prob_values = self.log_prob(xpos)

        # Convert log probabilities to actual PDF values
        pdf_values = torch.exp(log_prob_values)

        weights = pdf_values.T

        y = inputs @ weights

        return y
    


class RoamingGaussianDownsample1DNNStats(nn.Module):
    """
    learn a NN which outputs the gaussian stats as a function of the input.

    hmm it kinda seems like the gaus_nn could take on the same form as a UNet or transformer, and then it's like, well am I actually being any more efficient than just eschewing this layer altogether and simply using a transformer?
    """
    def __init__(self, 
                 in_w,
                 nb_gaussians,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_w = in_w
        self.nb_gaussians = nb_gaussians

        xpos = np.linspace(-1., 1., self.in_w)
        xpos2 = torch.from_numpy(xpos).float()

        self.xpos = torch.tile(xpos2, (self.nb_gaussians, 1))
        self.xpos = torch.unsqueeze(self.xpos, dim=0)

        n1 = 128
        self.gaus_nn = nn.Sequential(
            nn.Linear(in_w, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, n1),
            nn.LeakyReLU(),
            nn.Linear(n1, n1),
        )

        self.gaus_mean_head = nn.Sequential(
            nn.Linear(n1, nb_gaussians),
            nn.Tanh() # smooth, range [-1, 1] because we normalize pixel values to [-1, 1]
        )

        self.gaus_std_head = nn.Sequential(
            nn.Linear(n1, nb_gaussians),
            nn.Softplus() # smooth, no negative outputs
        )
        
    def log_prob(self, mean, std, value):
        # # Helper to calculate log probability of a value under the current distribution
        # distribution = Normal(mean, std)
        # return distribution.log_prob(value)
        loc = mean
        scale = std
        var = scale**2

        log_scale = (
            torch.log(scale)
        )

        y = (
            -((value - loc) ** 2) / (2 * var)
            - log_scale
            - np.log(np.sqrt(2 * np.pi))
        )
        return y

    def forward(self, inputs):
        b = inputs.shape[0]
        # get the x positions
        xpos = torch.tile(self.xpos, (b, 1, 1))

        # get means and stds from the nn
        gaus_features = self.gaus_nn(inputs)
        means = self.gaus_mean_head(gaus_features)
        stds = 0.1 * torch.ones([1, 1]) # self.gaus_std_head(gaus_features)


        means2 = torch.unsqueeze(means, dim=-1)
        stds2 = torch.unsqueeze(stds, dim=-1)

        means2.retain_grad()

        # Calculate log probability for each point
        log_prob_values = self.log_prob(means2, stds2, xpos)

        # Convert log probabilities to actual PDF values
        pdf_values = torch.exp(log_prob_values)

        weights = pdf_values
        weights.retain_grad()

        # add gaussian dim
        # x2 = torch.unsqueeze(inputs, dim=1)

        y = weights @ inputs.T
        y.retain_grad()

        y2 = torch.squeeze(y, dim=-1)
        y2.retain_grad()

        print(means2)

        return y2

def get_num_correct(preds, labels):
    # preds.argmax(dim=1) gets the index of the max log-probability/logit (the predicted class)
    # .eq(labels) checks where the predictions equal the actual labels, returning a boolean tensor
    # .sum() calculates the total number of correct predictions in the batch
    nb = preds.argmax(dim=1).eq(labels).sum().item()
    sr = nb / len(labels)
    return sr

def main():
    nb_inputs = 16
    nb_gaussians = 1

    nb_hidden = 16

    nb_outputs = 2
    
    # make a simple NN
    model = nn.Sequential(
        # 1. baseline
        # nn.Linear(nb_inputs, nb_gaussians),

        # 2. 
        # RoamingGaussianDownsample1DLearnableStats(nb_inputs, nb_gaussians),

        # 3. 
        RoamingGaussianDownsample1DNNStats(nb_inputs, nb_gaussians),

        nn.Linear(nb_gaussians, nb_hidden),
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