import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

PLOT = True

if PLOT:
    plt.ion()
    fig = plt.figure()
    fig2 = plt.figure()

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal

"""
idea:
each activation contains a probability of being activated vs turned off for a step, n.
The probability is initialized uniformly, probability could either by learned, or could be proportional to its impact on the output

1. learned: idk

2. proportional to impact: fcn of gradient w.r.t. output (not loss)

flavors:
adaptive dropout, bayesian dropout, MC dropout
    * bayesian dropout 

multi-headed attention
NAS RL - network something RL - controller network which proposes a network which is trained 

hypothesis:
start with fully connected. Over time, unproductive connections are used less often, hopefully leading to higher performance per avg neurons used.

During inference, could "compile" to remove all dead neurons

"""
BATCH_SIZE = 64
DEVICE = 'cpu'

transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Lambda(lambda x: torch.flatten(x)),
 ])

# 60k datapts
mnist = MNIST(".", download=True, transform=transform, train=True)


if PLOT:

    img = mnist[0][0].numpy()
    img = np.reshape(img, [-1,])

    img /= img.max()

    img = np.reshape(img, [28, 28])
    plt.imshow(img)
    plt.show()
    plt.colorbar()

data_loader = DataLoader(mnist,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                    )

    
# could update probabilities using saliency maps: https://debuggercafe.com/saliency-maps-in-convolutional-neural-networks/ 

# taking some code from here: https://github.com/NU-Haptics-Lab/diffusion_policy/blob/de8a2529925787c5e5d5842d0cbee51f27ee5d10/diffusion_policy/policy/dexnex_layers.py#L264

# gemini
class LearnableNormal(nn.Module):
    def __init__(self, ndim = 1):
        super(LearnableNormal, self).__init__()

        # Define the mean (mean) and standard deviation (std) as learnable parameters
        self.mean = nn.Parameter(
            torch.rand((ndim, 1)) * 2.0 - 1.0
            )

        # Ensure std is always positive by parameterizing its log, or using a constraint
        self.std = nn.Parameter(
            torch.rand((ndim, 1)) + 0.5
            )

    def log_prob(self, value):
        # Helper to calculate log probability of a value under the current distribution
        distribution = Normal(self.mean, self.std)
        return distribution.log_prob(value)
    
class RoamingGaussianDownsample1D(nn.Module):
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

        self.gaus = LearnableNormal(self.nb_gaussians)

    def forward(self, inputs):
        # get the x positions
        xpos = self.xpos

        # Calculate log probability for each point
        log_prob_values = self.gaus.log_prob(xpos)

        # Convert log probabilities to actual PDF values
        pdf_values = torch.exp(log_prob_values)

        weights = pdf_values.T

        y = inputs @ weights

        return y

def get_num_correct(preds, labels):
    # preds.argmax(dim=1) gets the index of the max log-probability/logit (the predicted class)
    # .eq(labels) checks where the predictions equal the actual labels, returning a boolean tensor
    # .sum() calculates the total number of correct predictions in the batch
    nb = preds.argmax(dim=1).eq(labels).sum().item()
    sr = nb / len(labels)
    return sr

def main():
    nb_inputs = 28 * 28
    nb_inputs_2 = int(nb_inputs/8)

    nb_hidden = nb_inputs_2
    nb_hidden_big = nb_inputs_2
    nb_hidden_small = nb_inputs_2 # 16

    nb_outputs = 10
    
    roaming_gaus = RoamingGaussianDownsample1D(nb_inputs, nb_gaussians=nb_inputs_2)

    # make a simple NN
    model = nn.Sequential(
        roaming_gaus,

        nn.Linear(nb_inputs_2, nb_inputs_2),
        nn.LeakyReLU(),

        # final layer to get outputs
        nn.Linear(nb_inputs_2, nb_outputs),
    )
    model = model.to(DEVICE)

    # optimizer - weight_decay should be small ~1e-3
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    # classification loss
    loss_fn = torch.nn.CrossEntropyLoss() # applies softmax to its input


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    metrics = {"avg_l": 0.0, "avg_sr": 0.0}

    nb_epochs = 10000
    nb_batches = 9999
    for _ in range(nb_epochs): # tqdm.tqdm(range(nb_epochs)):
        # reset


        with tqdm.tqdm(total = len(data_loader), postfix=metrics) as t:
            for idx, batch in enumerate(data_loader):
                inputs, labels = batch
                
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                if idx > nb_batches:
                    break

                opt.zero_grad()


                outputs = model(inputs)

                l = loss_fn(outputs, labels)
                l.backward()

                opt.step()

                # SR
                sr = get_num_correct(outputs, labels)

                # avg metrics
                metrics['avg_l'] = 0.99 * metrics['avg_l'] + 0.01 * float(l.detach())
                metrics['avg_sr'] = 0.99 * metrics['avg_sr'] + 0.01 * float(sr.detach())


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