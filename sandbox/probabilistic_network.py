import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt

PLOT = True

if PLOT:
    plt.ion()
    fig = plt.figure()
    fig2 = plt.figure()

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

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
DEVICE = 'cuda'

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

class ProbabilisticPass(nn.Module):
    """
    
    """
    def __init__(self, input_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_shape = input_shape

        # probabilities - higher is more likely
        self.probabilities = nn.Parameter(
            0.5 * torch.ones(self.input_shape)
        )
        
    def forward(self, inputs):
        self.inputs = inputs
        self.inputs.retain_grad()

        # if training
        if self.train:
            # get random values
            rands = torch.rand(self.input_shape)
            
            mask = rands < self.probabilities

        # if eval
        else:
            # only use the 50% most useful
            mask = self.probabilities > 0.5

        # save the mask
        self.mask = mask

        # output
        y = torch.zeros_like(inputs)
        y[:, mask] = inputs[:, mask]

        self.y = y
        self.y.retain_grad()
        return y
    
    @torch.no_grad()
    def update_probabilities(self):
        # compute stats
        grads = self.inputs.grad[:, self.mask]
        grads = torch.abs(grads)

        # remove the batch axis?
        grads = grads.mean(dim=0)

        # normalize
        m = grads.mean()
        std = grads.std()

        # 0.0 mean, 1 std
        grads = (grads - m) / std

        # map to [0, 1]
        grads = torch.sigmoid(grads)

        # hack for PoC
        alpha = 0.99
        self.probabilities[self.mask] = self.probabilities[self.mask] * alpha + (1 - alpha) * grads

        pass

    @torch.no_grad()
    def update_probabilities_2(self):
        """
        in this one, I pick a percentage of nodes I want to keep, and if you aren't in the top performing percent of nodes, I reduce your likelihood to zero
        """
        top_percent = 0.25

        if self.mask.sum() == 0:
            return

        # compute stats
        grads = self.inputs.grad[:, self.mask]
        grads = torch.abs(grads)

        # remove the batch axis?
        grads = grads.mean(dim=0)

        q = torch.quantile(grads, 1 - top_percent)
        top_percent = grads > q

        new_val = torch.zeros_like(self.probabilities[self.mask])

        new_val[top_percent] = 1.0

        # hack for PoC
        alpha = 0.99
        self.probabilities[self.mask] = self.probabilities[self.mask] * alpha + (1 - alpha) * new_val

        pass

class ElementWiseLinear(nn.Module):
    """
    not sure why this doesn't exist yet. Each input gets the opportunity to be scaled and biased
    """
    def __init__(self, 
                 input_size,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size

        self.A = nn.Parameter(
            torch.rand(self.input_size)
        )

        # self.B = nn.Parameter(
        #     torch.rand(self.input_size)
        # )

    def forward(self, inputs):
        x = inputs

        # element-wise mult
        x = x * self.A  # + self.B

        return x

class ProbabilisticDownsample(nn.Module):
    """
    from n input neurons, down-sample to m output neurons as a function of their productivity
    """
    def __init__(self, 
                 nb_inputs, 
                 nb_outputs,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        # self.productivity = nn.Parameter(
        #     torch.zeros((nb_inputs))
        # )
        self.productivity = torch.ones((nb_inputs), device=DEVICE)

        # self.register_full_backward_hook(self.module_backward_hook)

        self.indices = None

    def module_backward_hook(self, module, grad_input, grad_output):
        self.update()

    def forward(self, inputs):
        """
        """
        x = inputs
        b = x.shape[0]

        self.inputs = inputs
        # self.inputs.retain_grad()

        # compute probs
        probs = self.productivity.clone().detach()

        # s = probs.sum()
        # if not s == 0:
        #     probs /= probs.sum()
        # else:
        #     probs = torch.ones_like(probs) / self.nb_inputs
        
        # tile
        probs = torch.tile(probs, (b, 1))

        indices = torch.multinomial(probs, self.nb_outputs, replacement = False)

        # ensure indices are ordered smallest to largest (for consistency)
        indices2, _ = torch.sort(indices, dim=1)

        # save
        self.indices = indices2

        # index x
        y = torch.gather(x, 1, indices2)

        self.y = y
        # self.y.retain_grad()

        return y
    
    def update_2(self):
        """
        update the productivities via some metric
        """
        indices = self.indices
        # weights = self.y.grad.clone().detach() #type:ignore
        weights = self.y

        # take the abs val because we just want the pixels which have the biggest impact on all the outputs
        weights = np.abs(weights)

        b = indices.shape[0]

        p = torch.zeros((b, self.nb_inputs))

        # I want p[i][indices[i]] = grads[i] for all i
        p2 = torch.scatter(p, 1, indices, weights)

        # # init with original values
        # new_val = self.productivity.clone()

        # reduce along batch dim
        new_val = p2.sum(dim=0, dtype=torch.float) # by summing, I'm saying I want the frequency of being picked to also factor into it...

        # anything less than zero set to zero
        new_val[new_val < 0] = 0.0

        # only update productivity if it was tested
        mask = new_val > 0

        # hack for PoC
        alpha = 0.95
        self.productivity[mask] = self.productivity[mask] * alpha + (1 - alpha) * new_val[mask]


        pass
    
    def update(self):
        """
        update the productivities via activation val
        """
        if self.indices is None:
            return
        
        indices = self.indices
        # weights = self.y.grad.clone().detach() #type:ignore
        weights = self.inputs.clone().detach()

        if False:
            # # force it so that only large positive activation values are good
            # weights[weights < 0] = 0.0
            weights = torch.abs(weights)

            b = indices.shape[0]
            p = torch.zeros((b, self.nb_inputs))

            # I want p[i][indices[i]] = weights[i] for all i
            p2 = weights # torch.scatter(p, 1, indices, weights)

            # reduce along batch dim
            avg_activations = p2.sum(dim=0)

            # get the mask
            freq = torch.bincount(indices.flatten(), minlength=self.nb_inputs)
            mask = freq.nonzero().squeeze()

            new_val = torch.zeros(self.nb_inputs)
            new_val[mask] = avg_activations[mask]

            # scale the activations w.r.t. how many times they were used. Since weights are regularized to zero (default behavior of torch.optimizer I believe), over time we'd expect the activation of unused or small grad neurons to go to zero
            new_val[mask] /= freq[mask]


            # hack for PoC
            alpha = 0.95
            self.productivity[mask] = self.productivity[mask] * alpha + (1 - alpha) * new_val[mask]
        
        if True:
            # this is just max pooling, but with std instead of max as the function
            
            # encourage diversity in input value
            new_val = weights.std(dim=0)

            alpha = 0.95
            self.productivity = self.productivity * alpha + (1 - alpha) * new_val

        #debug
        if False:
            imgs = torch.reshape(new_val, (-1, 28, 28))
            imgs2 = torch.mean(imgs, dim=0)

            imgs3 = imgs2 / imgs2.max()

            imgs4 = imgs3.numpy()

            plt.figure
            plt.imshow(imgs4)

        # check edge activation values (they should all be zero...?)
        pass

    @torch.no_grad()
    def get_n_productive(self):
        vals, inds = torch.topk(self.productivity, self.nb_outputs)

        return vals, inds
    
    def get_productive_inputs(self):
        y = torch.zeros(self.nb_inputs, device=DEVICE)

        vals, inds = self.get_n_productive()

        y[inds] = vals

        return y
    
class StdDevPooling(nn.MaxPool1d):
    """
    just like max pooling, but instead of the max val within a window, computes the std dev for that window
    """
    def forward(self, inputs):
        # compute std dev along batch dim
        std = inputs.std(dim=1)
        
        
        
        pass
        

    
def module_backward_hook(module, grad_input, grad_output):
    # print(f"Module backward hook called for {module.__class__.__name__}")
    # print(f"Grad input: {grad_input}")
    # print(f"Grad output: {grad_output}")

    # Optionally return new grad_input and grad_output tuples
    module.update()
    # return grad_input, grad_output

def get_num_correct(preds, labels):
    # preds.argmax(dim=1) gets the index of the max log-probability/logit (the predicted class)
    # .eq(labels) checks where the predictions equal the actual labels, returning a boolean tensor
    # .sum() calculates the total number of correct predictions in the batch
    nb = preds.argmax(dim=1).eq(labels).sum().item()
    sr = nb / len(labels)
    return sr

def main():
    nb_inputs = 28 * 28
    nb_inputs_2 = int(nb_inputs/2)

    nb_hidden = 128
    nb_hidden_big = 128
    nb_hidden_small = 128 # 16


    nb_outputs = 10
    

    # pp = ProbabilisticPass(nb_hidden)
    # handle = pp.register_full_backward_hook(module_backward_hook)

    pds = ProbabilisticDownsample(nb_inputs, nb_inputs_2)
    # handle = pds.register_full_backward_hook(module_backward_hook)

    # make a simple NN
    model = nn.Sequential(
        # ElementWiseLinear(nb_inputs),
        # ElementWiseLinear(nb_inputs),
        # pds,

        nn.Linear(nb_inputs, 128),
        nn.LeakyReLU(),

        nn.Linear(128, 1024), # todo: multi-headed prob down-sample?
        # ProbabilisticDownsample(1024, 128),
        nn.LeakyReLU(),
        
        # # baseline -> works well, more params
        # nn.Linear(1024, 128),
        # nn.LeakyReLU(),
        
        # Test 1. max-pool -> works well, fewer params
        #   assumes that activation value is a proxy for how important a neuron is w.r.t. the output
        nn.MaxPool1d(kernel_size=int(1024/128)),



        # final layer to get outputs
        nn.Linear(128, 10),

        # nn.Linear(nb_inputs_2, nb_inputs_2),
        # ProbabilisticDownsample(nb_inputs_2, 128), # make sense to do before the activation I think... a bit more efficient I think
        # nn.LeakyReLU(),

        # nn.Linear(128, 256),
        # ProbabilisticDownsample(256, 64),
        # nn.LeakyReLU(),

        # nn.Linear(64, 128),
        # ProbabilisticDownsample(128, 32),
        # nn.LeakyReLU(),

        # nn.Linear(32, 64),
        # ProbabilisticDownsample(64, 16),
        # nn.LeakyReLU(),

        # nn.Linear(16, 32),
        # ProbabilisticDownsample(32, 10),
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

                # compute the grad w.r.t. the output to see what inputs affected the output the most strongly
                # o2 = outputs.sum()
                # o2.backward(retain_graph=True)

                # update the productivity
                pds.update()

                l = loss_fn(outputs, labels)
                opt.zero_grad()
                l.backward()

                opt.step()

                # SR
                sr = get_num_correct(outputs, labels)

                # avg metrics
                metrics['avg_l'] = 0.99 * metrics['avg_l'] + 0.01 * float(l)
                metrics['avg_sr'] = 0.99 * metrics['avg_sr'] + 0.01 * float(sr)


                t.set_postfix(metrics)
                t.update()

        # end of epoch
        # print(pp.probabilities)

        model.eval()

        # validation

        # reset to training mode
        model.train()

        if True:
            # p = pds.productivity.clone().detach()
            p = pds.get_productive_inputs()

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