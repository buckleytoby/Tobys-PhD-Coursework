import numpy as np
import copy
import tqdm
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

PLOT = False

if PLOT:
    plt.ion()
    fig = plt.figure()

import torch
import torch.utils
import torch.utils.data
import cProfile, pstats

from torch.utils.data import TensorDataset, DataLoader
from torch.profiler import profile, ProfilerActivity, record_function

"""
min example


lit review:
logic gate networks
https://arxiv.org/pdf/2210.08277

sparse neural networks
https://medium.com/aimonks/sparse-neural-networks-and-pruning-trimming-the-fat-for-efficient-machine-learning-5b9d2920c526
"""
# globals
DEVICE = "cuda"

BATCH_SIZE = 10
NB_CHANGES_PER_ITER = 2
MAX_NB_FORWARD_PASSES = 5
NB_FREE_NODES = 10
PRINT_FREQ = 100
NB_ITERS = 200

# https://stackoverflow.com/questions/50052295/how-do-you-load-mnist-images-into-pytorch-dataloader
transform = transforms.Compose(
[transforms.ToTensor(),])

mnist = MNIST(".", download=True, transform=transform, train=True)

data_loader = torch.utils.data.DataLoader(mnist,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                        )

def post_process_sample(x):
    y = torch.zeros_like(x, dtype=torch.bool)

    # convert to binary
    y[x > 0] = True

    return y

def encode(x: int):
    # to boolean array
    ff = format(x, '08b') # 8 bits

    # make bool array
    # y = torch.array([True if f=="1" else False for f in ff])

    # gemini
    int_array = np.fromiter(ff, dtype=np.int8)
    boolean_array = int_array.astype(bool)

    y = boolean_array
    # assert(y.ndim == 2)
    # assert(y.shape[0] == 1)

    return y

def encode_arr(x):
    # vec_fcn = torch.vectorize(encode)
    # y = vec_fcn(x)
    y = np.vstack([encode(xx) for xx in x])


    return y

def decode_classification(x):
    """
    order doesn't matter, just must be consistent
    """
    # convert to int8
    x2 = x.to(torch.int8)
    
    # take the first true value
    i = torch.argmax(x2, dim=1)

    return i

def decode(x):
    bit_array = x
    # gemini
    N = bit_array.shape[1]
    weights = 2 ** torch.arange(N)[::-1] # [128, 64, 32, 16, 8, 4, 2, 1]

    # Multiply the bits by their weights
    weighted_bits = bit_array * weights

    # Sum the results to get the final integer
    integer_value = torch.sum(weighted_bits, dim=1)

    # end gemini

    return integer_value

def nand(a, b):
    """
    shape = batch, nb bits
    """
    # array and, along bit dimension
    a2 = torch.logical_and(a, b)

    # array nand
    a3 = torch.logical_not(a2)

    return a3

def identitya(a, b):
    # let a through
    return a

class Node:
    def __init__(self, 
                 batch_size,
                 default_x1,
                 default_x2,
                 settable = True, # whether its daddy's are allowed to be changed
                 fcn = nand,
                 descr = "",
                 id = 0,
                 im_a_copy = False,
                 ): # , input_shape):
        # self.input_shape = input_shape
        self.batch_size = batch_size
        self.settable = settable
        self.fcn = fcn
        self.descr = descr
        self.id = id

        # my members

        # if default inputs are 0,0 then default output will be 1
        self.default_x1 = default_x1 # torch.zeros([batch_size, 1], dtype=torch.bool)
        self.default_x2 = default_x2 # torch.zeros([batch_size, 1], dtype=torch.bool)

        # self.daddy1: None | Node = None
        # self.daddy2: None | Node = None

        self.daddy_id1: None | int = None
        self.daddy_id2: None | int = None

        if not im_a_copy:
            self.reset()

            # forward once to save a y-value
            # self.forward(self.x1, self.x2)

            # self.save_self()

    def reset(self):
        self.x1 = self.default_x1
        self.x2 = self.default_x2

        self.forward(self.x1, self.x2)

    def copy(self):
        n = Node(
            self.batch_size,
            self.default_x1,
            self.default_x2,
            self.settable,
            self.fcn,
            self.descr,
            self.id,
            im_a_copy=True,
        )

        # copy my members
        n.daddy_id1 = self.daddy_id1
        n.daddy_id2 = self.daddy_id2

        return n
    
    def revert(self):
        """
        revert to my old self
        """
        if self.settable:
            self.daddy_id1 = self.old_self.daddy_id1
            self.daddy_id2 = self.old_self.daddy_id2

    def save_self(self):
        """
        save myself as my old self
        """
        self.old_self = self.copy()
        pass

    def get_daddies(self, old):
        if not old:
            return self.daddy_id1, self.daddy_id2
        
        else:
            return self.old_self.daddy_id1, self.old_self.daddy_id2
    
    def forward(self, x1, x2):
        # self.update()
        self.x1 = x1
        self.x2 = x2

        self.y = self.fcn(self.x1, self.x2)

        assert(self.y.shape == (self.batch_size, 1))
        return self.y
    
    # def copy(self):
    #     n = Node(
    #         self.batch_size,
    #         self.settable,
    #         self.fcn,
    #         self.descr
    #     )

    def nb_edges(self, old):
        id1, id2 = self.get_daddies(old)

        nb = 0
        nb += id1 is not None
        nb += id2 is not None
        return nb
    
class Change:
    def __init__(
            self,
            tm,
            node1: None | Node,
            node2: None | Node,
            which_one,
    ):
        self.tm = tm
        self.node1 = node1
        self.node2 = node2
        self.which_one = which_one

    def set_daddy1(self):
        assert(isinstance(self.node1, Node))
        assert(isinstance(self.node2, Node))

        self.node2.daddy_id1 = self.node1.id

    def set_daddy2(self):
        assert(isinstance(self.node1, Node))
        assert(isinstance(self.node2, Node))

        self.node2.daddy_id2 = self.node1.id

    def new_node(self):
        self.tm.new_node()

    def sever_edge_1(self):
        assert(isinstance(self.node2, Node))
        self.node2.daddy_id1 = None

    def sever_edge_2(self):
        assert(isinstance(self.node2, Node))
        self.node2.daddy_id2 = None

    def execute(self):
        """
        
        """
        # set daddy of node2 to node1
        match self.which_one:
            case 0:
                self.set_daddy1()

            case 1:
                self.set_daddy2()

            case 2:
                self.new_node()

            case 3:
                self.sever_edge_1()

            case 4:
                self.sever_edge_2()

def false_mask(nb_nodes):
    return torch.zeros([nb_nodes], dtype=torch.bool, device=DEVICE)

def zero_index(nb_nodes):
    return torch.zeros([nb_nodes], dtype=torch.int, device=DEVICE)

class State:
    def __init__(self,
                 batch_size,
                 nb_nodes,
                 ) -> None:
        self.batch_size = batch_size
        self.nb_nodes = nb_nodes

        # make the default arrays
        self.x1_default = torch.zeros([batch_size, nb_nodes], dtype=torch.bool, device=DEVICE)
        self.x2_default = torch.zeros([batch_size, nb_nodes], dtype=torch.bool, device=DEVICE)

        ## make all the empty data arrays, bool
        self.x1_arr = torch.zeros([batch_size, nb_nodes], dtype=torch.bool, device=DEVICE)
        self.x2_arr = torch.zeros_like(self.x1_arr, dtype=torch.bool, device=DEVICE)
        self.y_arr = torch.zeros_like(self.x1_arr, dtype=torch.bool, device=DEVICE)

        # bool masks?
        self.input_mask = false_mask(nb_nodes)
        self.output_mask = false_mask(nb_nodes)
        self.identitya_mask = false_mask(nb_nodes)
        self.nand_mask = false_mask(nb_nodes)
        self.x1_daddy_mask = false_mask(nb_nodes)
        self.x2_daddy_mask = false_mask(nb_nodes)
        self.settable_mask = false_mask(nb_nodes)

        # indices arrays
        self.x1_daddies = zero_index(nb_nodes)
        self.x2_daddies = zero_index(nb_nodes)
        
    def setup(self, nb_inputs, nb_outputs):
        """
        order matters
        """
        # must make all bool masks first
        i = 0
        
        # all inputs - input-mask, identity-a
        self.input_mask[i:i+nb_inputs] = True
        self.identitya_mask[i:i+nb_inputs] = True
        
        i += nb_inputs
        # all outputs - output-mask, nand-mask, x1-daddy-mask, x2-daddy-mask, settable-mask
        self.output_mask[i:i+nb_outputs] = True
        self.nand_mask[i:i+nb_outputs] = True
        self.x1_daddy_mask[i:i+nb_outputs] = True
        self.x2_daddy_mask[i:i+nb_outputs] = True
        self.settable_mask[i:i+nb_outputs] = True
        
        i += nb_outputs
        # free nodes - nand-mask, x1-daddy-mask, x2-daddy-mask, settable-mask
        self.nand_mask[i:] = True
        self.x1_daddy_mask[i:] = True
        self.x2_daddy_mask[i:] = True
        self.settable_mask[i:] = True
        
        # then, can randomize the edges
        self.randomize_edges()
        
        pass

    def get_nb_edges(self):
        nb = 0
        nb += self.x1_daddy_mask.sum()
        nb += self.x2_daddy_mask.sum()

        return nb
    
    def array_forward_reset(self):
        self.x1 = self.x1_default.clone()
        self.x2 = self.x2_default.clone()

        self.array_forward_one_pass()

    def copy(self):
        s = State(self.batch_size, self.nb_nodes)

        s.x1_arr = self.x1_arr.clone()
        s.x2_arr = self.x2_arr.clone()

        s.input_mask = self.input_mask.clone()
        s.output_mask = self.output_mask.clone()
        s.identitya_mask = self.identitya_mask.clone()
        s.nand_mask = self.nand_mask.clone()
        s.x1_daddy_mask = self.x1_daddy_mask.clone()
        s.x2_daddy_mask = self.x2_daddy_mask.clone()
        s.settable_mask = self.settable_mask.clone()

        s.x1_daddies = self.x1_daddies.clone()
        s.x2_daddies = self.x2_daddies.clone()

        return s

    def array_set_input(self, input):
        """
        set inputs
        """
        self.x1_arr[:, self.input_mask] = input
        pass
    
    # @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def array_forward_one_pass(self):
        # inherit from daddy, use broadcasting? or tiling? fancy indexing?
        self.x1_arr[:, self.x1_daddy_mask] = self.y_arr[:, self.x1_daddies[self.x1_daddy_mask]]
        self.x2_arr[:, self.x2_daddy_mask] = self.y_arr[:, self.x2_daddies[self.x2_daddy_mask]]

        # apply fcn masks to x1 and x2 to produce y
        self.y_arr[:, self.identitya_mask] = identitya(self.x1_arr[:, self.identitya_mask], self.x2_arr[:, self.identitya_mask])
        self.y_arr[:, self.nand_mask] = nand(self.x1_arr[:, self.nand_mask], self.x2_arr[:, self.nand_mask])
        
        return self.y_arr

    def get_output(self):
        return self.y_arr[:, self.output_mask]
    
    def get_input(self):
        return self.x1_arr[:, self.input_mask]

    def randomize_x(self, x, mask):
        """
        perturb nb edges of the settable set
        """
        N = mask.sum()

        nb = int(0.1 * N)

        idxs = torch.randint(0, N, (nb,), device=DEVICE)

        # any node can be a daddy
        new_daddies = torch.randint(0, self.nb_nodes, (nb,), device=DEVICE)

        # idxs2d = torch.unravel_index(idxs, self.x1_daddies[mask])
        old_daddies = x[mask]
        old_daddies[idxs] = new_daddies

        x[mask] = old_daddies

        pass
    
    def randomize_edge(self, daddies):
           
        nb = self.settable_mask.sum()
        new_daddies = torch.randint(0, self.nb_nodes, (nb,), dtype=torch.int, device=DEVICE)
        
        daddies[self.settable_mask] = new_daddies
        
        pass
    
    def randomize_edges(self):
        """
        
        """
        self.randomize_edge(self.x1_daddies)
        self.randomize_edge(self.x2_daddies)
        
        
    
class TuringMachine:
    def __init__(self, batch_size, nb_inputs, nb_outputs, im_a_copy=False):
        self.batch_size = batch_size
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        
        # self.initialize_nodes()
        nb_nodes = self.nb_inputs + self.nb_outputs + NB_FREE_NODES

        self.state = State(self.batch_size, nb_nodes)

        self.state.setup(self.nb_inputs, self.nb_outputs)
            
    def initialize_nodes(self):
        # # I/O
        # self.inputs = [Node(batch_size, default_false, default_false, settable=False, fcn=identitya, descr="input") for _ in range(nb_inputs)]
        # self.outputs = [Node(batch_size, default_false, default_false, descr="output") for _ in range(nb_outputs)]

        # # extra nodes
        # self.nodes = [self.HALT, self.TRUE, self.FALSE] + self.inputs + self.outputs

        # # available to be used
        # nb = NB_FREE_NODES
        # self.nodes += [Node(batch_size, default_false, default_false) for _ in range(nb)]
        pass
    
    def get_nb_nodes(self):
        return self.state.nb_nodes

    def perturb_x(self, x, mask):
        """
        perturb nb edges of the settable set
        """
        N = mask.sum()

        nb = int(0.1 * N)

        idxs = torch.randint(0, N, (nb,), dtype=torch.int, device=DEVICE)

        # any node can be a daddy
        new_daddies = torch.randint(0, self.get_nb_nodes(), (nb,), dtype=torch.int, device=DEVICE)

        # idxs2d = torch.unravel_index(idxs, self.x1_daddies[mask])
        old_daddies = x[mask]
        old_daddies[idxs] = new_daddies

        x[mask] = old_daddies

        pass

    def perturb_x1(self):
        """
        perturb nb edges of the settable set
        """
        self.perturb_x(self.state.x1_daddies, self.state.x1_daddy_mask)

    def perturb_x2(self):
        """
        perturb nb edges of the settable set
        """
        self.perturb_x(self.state.x2_daddies, self.state.x2_daddy_mask)

    def array_forward_reset(self):
        """
        reset to default
        """
        

    def array_set_input(self, input):
        """
        set inputs
        """
        self.state.array_set_input(input)
        pass

    def array_forward_one_pass(self):
        self.state.array_forward_one_pass()

    def get_output(self):
        return self.state.get_output()
    
    def get_input(self):
        return self.state.get_input()

    def forward(self, input, old=False):
        done = False
        max_iter = MAX_NB_FORWARD_PASSES
        c = 0

        if old:
            state = self.old_state
        else:
            state = self.state

        state.array_forward_reset()
        state.array_set_input(input)
        output = state.get_output()

        while not done:
            c += 1

            state.array_forward_one_pass()

            # check the halt bit
            # done = done or torch.all(state.HALT.y)

            # 
            new_output = state.get_output()

            # no change in output?
            no_change = torch.all(output == new_output)

            # if no change
            if no_change:
                done = True

            # if change
            else:
                # save output
                output = new_output

            # check iter timeout
            if c > max_iter:
                done = True

        # return final output
        y = state.get_output()

        return y
    
    def revert(self):
        """
        revert all node connects to their old state
        """
        self.state = self.old_state

        # now save a copy so I can change self.state without issue
        self.save_self()
        pass

    def save_self(self):
        """
        save my state
        """
        self.old_state = self.state.copy()
        pass
    
    def get_nb_edges(self, old=False):
        # nb = torch.sum([n.nb_edges(old) for n in self.nodes])
        nb = self.state.get_nb_edges()

        return nb
    
def binary_diff(x, y):
    right = x == y
    wrong = torch.logical_not(right)

    l1 = torch.mean(torch.abs(wrong))
    return l1

def mse(x, y):
    l = 0.5 * torch.abs(x - y) ** 2
    l2 = torch.mean(l)

    return l2

def one_hot(x, y):
    """
    classification tasks. assumes x, y are int arrays
    """
    right = x == y
    wrong = torch.logical_not(right)

    # wrong2 = wrong.to(torch.int8)

    l1 = torch.mean(wrong, dtype=torch.float)
    return l1

def val(x, yintgt, tm):
    """
    classification val
    """
    yb = tm.forward(x)
    yint = decode_classification(yb)

    l1 = one_hot(yint, yintgt)
    return l1

    
def loss(x, yintgt, ybgt, tm: TuringMachine, old):
    """
    classification loss
    """
    yb = tm.forward(x, old)
    yint = decode_classification(yb)

    l1 = one_hot(yint, yintgt)

    # l1 = one_hot(yint, ygt)
    # l1 = binary_diff(yb, ybgt)

    # add on size cost
    w = tm.get_nb_edges(old) / 10000.0
    l2 = 0.0

    # weigh by size --> larger size means larger loss
    l3 = l1 + l2

    return l1, l3

def convert_entire_dataset():
    print("converting entire dataset. Should only happen once.")
    xbs = []
    ybs = []
    yints = []

    for idx, (x, y) in enumerate(data_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        xb = post_process_sample(x)

        # xb = encode(x) # not needed after post_process_sample
        xb = torch.reshape(xb, xb.shape[0:1] + (-1,))
        yb = encode_arr(y)

        xb = torch.tensor(xb)
        yb = torch.tensor(yb)

        # y = y.numpy()
        xbs.append(xb)
        ybs.append(yb)
        yints.append(y)

    xbs_t = torch.vstack(xbs)
    ybs_t = torch.vstack(ybs)
    yints_t = torch.flatten(torch.vstack(yints))

    tds = TensorDataset(xbs_t, yints_t, ybs_t)

    print("Done converting dataset")
    return tds

dataset = convert_entire_dataset()

def get_batch():
    global dataset

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            )
    
    while True:
        for idx, (xb, yint, yb) in enumerate(data_loader):
            # xb = post_process_sample(x)

            # # xb = encode(x) # not needed after post_process_sample
            # xb = torch.reshape(xb, xb.shape[0:1] + (-1,))
            # yb = encode_arr(y)

            # y = y.numpy()
            # yield xb, y, yb
            # xb = xb.numpy()
            # yint = yint.numpy()
            # yb = yb.numpy()
            yield xb, yint, yb



@torch.no_grad()
def main():
    # x = D[:, 0:1]
    # y = D[:, 1:2]
    batch_gen = get_batch()

    # xb = torch.array([encode(x2) for x2 in x])
    # yb = torch.array([encode(y2) for y2 in y])
    xb, yint, ybgt = next(batch_gen)

    # input node
    input_shape = (2, 1)
    output_shape = (2, 1)
    nb_inputs = 28*28
    nb_outputs = 10 # must be nb classes for classification (10 for mnist)
    tm = TuringMachine(BATCH_SIZE, nb_inputs, nb_outputs)

    # test
    gt_loss, l = loss(xb, yint, ybgt, tm, old=False)
    print("default l", l)
    
    done = False
    eps = 1e-2

    hash = {}
    min_avg_loss = 0
    c = 0
    every = 1
    # old_tm = tm.clone()
    # new_tm = old_tm.clone()
    tm.save_self()

    # history
    h = []

    # loop vars
    avg_l = 0
    old_avg_l = 0

    while not done:
        c += 1

        # reset loop vars
        avg_l = 0
        old_avg_l = 0

        # sanity check
        if False:
            print("\noriginal")
            print(loss(xb, yint, ybgt, tm, old=True))
            print(loss(xb, yint, ybgt, tm, old=False))
            pass

        # make random changes
        for _ in range(NB_CHANGES_PER_ITER):
            # changes = tm.changes # hashed

            # randomly select some changes
            # nb_changes = torch.random.randint(0, int(0.1 * len(changes)))

            tm.perturb_x1()
            tm.perturb_x2()
            # ii = torch.random.randint(0, len(changes), nb_changes)

            # # get it 
            # for i in ii:
            #     change: Change = changes[i]

            #     # execute it
            #     change.execute()

        # recompile -- not needed since I moved everything to arrays
        # tm.compile()

        # test old a new batch loss. If you want better stats, increase batch_size (rather than putting this section in a for-loop)
        # new batch
        # xb, yint, ybgt = next(batch_gen)

        # copy
        # new_tm = tm.clone()

        # collate all possible changes
        # changes = tm.get_all_changes()

        # get loss
        gt_loss, l = loss(xb, yint, ybgt, tm, old=False)
        avg_l += l / every

        gt_loss, l = loss(xb, yint, ybgt, tm, old=True)
        old_avg_l += l / every
        # print("loss: ", l)

        # sanity check
        if False:
            print("post-change. shouldn't be any change to old=True")
            print(loss(xb, yint, ybgt, tm, old=True))
            print(loss(xb, yint, ybgt, tm, old=False))
            pass


        # save?
        if True: # c%every == 0:
            minl = torch.min(avg_l, old_avg_l)
            
            min_avg_loss += minl

            # if avg_l < old_avg_l:
            #     print("avg loss improved! to: ", avg_l)

            # changes were good. Stay with new_tm
            if avg_l < old_avg_l:
                # min_avg_loss = avg_l
                # tm = new_tm

                # save new_tm to old_tm
                # old_tm = new_tm
                tm.save_self()
                pass

                # sanity check
                if False:
                    print("save")
                    print(loss(xb, yint, ybgt, tm, old=True))
                    print(loss(xb, yint, ybgt, tm, old=False))
                    pass

            # changes were no good. Revert.
            else:
                # reset new_tm to old tm
                # new_tm = old_tm.clone()
                tm.revert()

                # sanity check
                if False:
                    print("revert")
                    print(loss(xb, yint, ybgt, tm, old=True))
                    print(loss(xb, yint, ybgt, tm, old=False))
                    pass

        if c%PRINT_FREQ == 0:
            print("---------")
            print("old avg loss: ", old_avg_l)
            print("avg loss: ", avg_l)
            print("min avg loss", min_avg_loss / PRINT_FREQ)
            print("iter: ", c)
            print("nb edges: ", tm.get_nb_edges())
            print("gt loss: ", gt_loss)
            print("one-hot val loss: ", val(xb, yint, tm))
            # print("gt: ", yb)
            # print("output: ", new_tm.get_output())

            # history
            h.append(min_avg_loss / PRINT_FREQ)

            if PLOT:
                plt.clf()
                plt.plot(h[-20:])
                plt.show()
                fig.canvas.draw()
                fig.canvas.flush_events()

            # reset
            min_avg_loss = 0

        # early exit
        if old_avg_l < eps:
            done = True

        if c > NB_ITERS:
            done = True

        pass
    
    print("final stats")
    print("iter: ", c)
    print("nb edges: ", tm.get_nb_edges())
    # print("gt: ", yb)
    # print("output: ", old_tm.get_output())


    pass

if True:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        main()
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if False:
    # gemini
    # 1. Instantiate the profiler object
    profiler = cProfile.Profile()

    profiler.run("main()")

    # 3. Save the raw statistics to a file
    raw_stats_file = "my_profile_data.prof"
    profiler.dump_stats(raw_stats_file)

    # 4. Load the raw stats from the file
    stats = pstats.Stats(raw_stats_file)

    # Optional: Sort the statistics (e.g., by cumulative time)
    stats.sort_stats('time')

    # 5. Redirect the print output to a text file
    output_txt_file = "cprofile_report.txt"

    with open(output_txt_file, 'w') as f:
        # Set the stream to the file object 'f'
        stats.stream = f #type:ignore
        
        # Print the formatted statistics to the file
        stats.print_stats()
        
    print(f"Profiling report successfully saved to {output_txt_file}")
    
    
if False:
    main()