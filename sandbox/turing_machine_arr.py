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
import cProfile, pstats

from torch.utils.data import TensorDataset, DataLoader

"""
min example


lit review:
logic gate networks
https://arxiv.org/pdf/2210.08277

sparse neural networks
https://medium.com/aimonks/sparse-neural-networks-and-pruning-trimming-the-fat-for-efficient-machine-learning-5b9d2920c526
"""
# globals

BATCH_SIZE = 10
NB_CHANGES_PER_ITER = 2
MAX_NB_FORWARD_PASSES = 10
NB_FREE_NODES = 5000
PRINT_FREQ = 50
NB_ITERS = 200000

# https://stackoverflow.com/questions/50052295/how-do-you-load-mnist-images-into-pytorch-dataloader
transform = transforms.Compose(
[transforms.ToTensor(),])

mnist = MNIST(".", download=True, transform=transform, train=True)

data_loader = torch.utils.data.DataLoader(mnist,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                        )

def post_process_sample(x):
    y = np.zeros_like(x, dtype=np.bool_)

    # convert to binary
    y[x > 0] = True

    return y


D = np.array(
    [[0, 1],
     [1, 2],
     [2, 3],
     [3, 0]]
)

def encode(x: int):
    # to boolean array
    ff = format(x, '08b') # 8 bits

    # make bool array
    # y = np.array([True if f=="1" else False for f in ff])

    # gemini
    int_array = np.fromiter(ff, dtype=np.int8)
    boolean_array = int_array.astype(bool)

    y = boolean_array
    # assert(y.ndim == 2)
    # assert(y.shape[0] == 1)

    return y

def encode_arr(x: np.ndarray):
    # vec_fcn = np.vectorize(encode)
    # y = vec_fcn(x)
    y = np.vstack([encode(xx) for xx in x])


    return y

def decode_classification(x):
    """
    order doesn't matter, just must be consistent
    """
    # take the first true value
    i = np.argmax(x, axis=1)

    return i

def decode(x):
    bit_array = x
    # gemini
    N = bit_array.shape[1]
    weights = 2 ** np.arange(N)[::-1] # [128, 64, 32, 16, 8, 4, 2, 1]

    # Multiply the bits by their weights
    weighted_bits = bit_array * weights

    # Sum the results to get the final integer
    integer_value = np.sum(weighted_bits, axis=1)

    # end gemini

    return integer_value

def decode_arr(x):
    y = np.vstack([encode(xx) for xx in x])

def nand(a, b):
    """
    shape = batch, nb bits
    """
    # array and, along bit dimension
    a2 = np.logical_and(a, b)

    # array nand
    a3 = ~a2

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
        self.default_x1 = default_x1 # np.zeros([batch_size, 1], dtype=np.bool_)
        self.default_x2 = default_x2 # np.zeros([batch_size, 1], dtype=np.bool_)

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

    # def fjsdklfsforward(self, x):
    #     x1 = x[:, 0:1]
    #     x2 = x[:, 1:2]

    #     y1 = nand(x1, x1)
    #     y2 = nand(x1, x2)
    #     y3 = nand(x2, x2)

    #     # y = np.concat([y1, y2, y3], axis=1)

    #     z1 = nand(y1, y2)
    #     z2 = nand(y2, y3)

    #     z = np.concat([z1, z2], axis=1)

    #     return z
    
    # def forward2(self, x1, x2):
    #     y = nand(x1, x2)
    #     return y

    # def update(self):
    #     if self.daddy1 is not None:
    #         self.x1 = self.daddy1.y

    #     if self.daddy2 is not None:
    #         self.x2 = self.daddy2.y
    
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
    return np.zeros([nb_nodes], dtype=np.bool_)

def zero_index(nb_nodes):
    return np.zeros([nb_nodes], dtype=np.int_)

class State:
    def __init__(self,
                 batch_size,
                 nb_nodes,
                 ) -> None:
        self.batch_size = batch_size
        self.nb_nodes = nb_nodes

        # make the default arrays
        self.x1_default = np.zeros([batch_size, nb_nodes])
        self.x2_default = np.zeros([batch_size, nb_nodes])

        ## make all the empty arrays
        self.x1_arr = np.zeros([batch_size, nb_nodes])
        self.x2_arr = np.zeros_like(self.x1_arr)
        self.y_arr = np.zeros_like(self.x1_arr)

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

    def get_nb_edges(self):
        nb = 0
        nb += self.x1_daddy_mask.sum()
        nb += self.x2_daddy_mask.sum()

        return nb
    
    def array_forward_reset(self):
        self.x1 = self.x1_default.copy()
        self.x2 = self.x2_default.copy()

        self.array_forward_one_pass()

    def copy(self):
        s = State(self.batch_size, self.nb_nodes)

        s.x1_arr = self.x1_arr.copy()
        s.x2_arr = self.x2_arr.copy()

        s.input_mask = self.input_mask.copy()
        s.output_mask = self.output_mask.copy()
        s.identitya_mask = self.identitya_mask.copy()
        s.nand_mask = self.nand_mask.copy()
        s.x1_daddy_mask = self.x1_daddy_mask.copy()
        s.x2_daddy_mask = self.x2_daddy_mask.copy()
        s.settable_mask = self.settable_mask.copy()

        s.x1_daddies = self.x1_daddies.copy()
        s.x2_daddies = self.x2_daddies.copy()

        return s

    def array_set_input(self, input):
        """
        set inputs
        """
        self.x1_arr[:, self.input_mask] = input
        pass
        
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
    
class TuringMachine:
    def __init__(self, batch_size, nb_inputs, nb_outputs, im_a_copy=False):
        self.batch_size = batch_size
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        # defaults
        default_false = np.zeros([batch_size, 1], dtype=np.bool_)
        default_true = np.ones([batch_size, 1], dtype=np.bool_)

        self.HALT = Node(batch_size, default_true, default_true, descr="HALT")

        # true rail
        self.TRUE = Node(batch_size, default_true, default_true, settable=False, fcn=identitya, descr="TRUERAIL")

        # false rail
        self.FALSE = Node(batch_size, default_false, default_false, settable=False, fcn=identitya, descr="FALSERAIL")

        # I/O
        self.inputs = [Node(batch_size, default_false, default_false, settable=False, fcn=identitya, descr="input") for _ in range(nb_inputs)]
        self.outputs = [Node(batch_size, default_false, default_false, descr="output") for _ in range(nb_outputs)]

        # extra nodes
        self.nodes = [self.HALT, self.TRUE, self.FALSE] + self.inputs + self.outputs

        # available to be used
        nb = NB_FREE_NODES
        self.nodes += [Node(batch_size, default_false, default_false) for _ in range(nb)]

        if not im_a_copy:
            self.update_ids()

            # randomly initialize connections
            self.randomize_connections()

            # index all changes
            self.get_all_changes()

            # save self
            # self.save_self()

            self.compile()

    def compile(self):
        """
        turn the class structure into a flat array structure

        output:
        data-arrays
        self.x1_arr = array of x1 values, b x nb_nodes
        self.x2_arr = array of x2 values, b x nb_nodes
        self.y_arr = output array, b x nb_nodes

        masks
        self.x1_mask = 1 x nb_nodes
        self.x2_mask = 1 x nb_nodes

        """
        nb_nodes = self.get_size()

        def false_mask():
            return np.zeros([nb_nodes], dtype=np.bool_)
        
        def zero_index():
            return np.zeros([nb_nodes], dtype=np.int_)

        self.state = State(self.batch_size, nb_nodes)

        # iterate through nodes
        for node in self.nodes:
            if node.descr == "input":
                self.state.input_mask[node.id] = True

            elif node.descr == "output":
                self.state.output_mask[node.id] = True

            if node.fcn == identitya:
                self.state.identitya_mask[node.id ] = True

            elif node.fcn == nand:
                self.state.nand_mask[node.id ] = True

            else:
                raise
                
            if node.daddy_id1 is not None:
                self.state.x1_daddy_mask[node.id] = True
                self.state.x1_daddies[node.id] = node.daddy_id1
                
            if node.daddy_id2 is not None:
                self.state.x2_daddy_mask[node.id] = True
                self.state.x2_daddies[node.id] = node.daddy_id2

            self.state.settable_mask[node.id] = node.settable

        pass

    def perturb_x(self, x, mask):
        """
        perturb nb edges of the settable set
        """
        N = mask.sum()

        nb = int(0.1 * N)

        idxs = np.random.randint(0, N, nb)

        # any node can be a daddy
        new_daddies = np.random.randint(0, self.get_nb_nodes(), nb)

        # idxs2d = np.unravel_index(idxs, self.x1_daddies[mask])
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

    def update_ids(self):
        node: Node
        for idx, node in enumerate(self.nodes):
            node.id = idx

    def register_new_node(self, n: Node):
        self.nodes.append(n)

        n.id = len(self.nodes) - 1

        n.save_self()

        # re-index all available changes
        self.get_all_changes()
        pass

    def randomize_connection(self, node):
        nb_nodes = len(self.nodes)

        node2 = node
        if node2.settable:
            for which_one in [0, 1]:
                i = np.random.randint(0, nb_nodes)
                
                node1 = self.nodes[i]

                c = Change(self, node1, node2, which_one)
                c.execute()

                if node2.descr == "output":
                    pass

    def randomize_connections(self):

        for node2 in self.nodes:
            self.randomize_connection(node2)

    def new_node(self):
        default_false = np.zeros([self.batch_size, 1], dtype=np.bool_)
        n = Node(self.batch_size, default_false, default_false)
        self.randomize_connection(n)

        self.register_new_node(n)


        pass

    def get_output(self):
        return self.state.get_output()
    
    def get_input(self):
        return self.state.get_input()

    def set_inputs(self, input):
        nb_inputs = input.shape[1]
        for idx in range(nb_inputs):
            self.inputs[idx].x1 = input[:, idx:idx+1]

    def get_node_inputs(self, node: Node, old):
        # id1 = node.daddy_id1
        # id2 = node.daddy_id2
        id1, id2 = node.get_daddies(old)

        if id1 is not None:
            x1 = self.nodes[id1].y
        else:
            x1 = node.x1

        if id2 is not None:
            x2 = self.nodes[id2].y
        else:
            x2 = node.x2

        return x1, x2
    
    def reset(self):
        for node in self.nodes:
            node.reset()

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
            # done = done or np.all(state.HALT.y)

            # 
            new_output = state.get_output()

            # no change in output?
            no_change = np.all(output == new_output)

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

    def get_all_changes(self):
        """
        1-step changes
        """
        changes = []

        # add output if i to input n of j, i can be == j
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node2.settable:
                    # only change 1 or 2 aka set daddy 1 or 2, 3 or 4 is sever daddy 1 or 2
                    for i in [0, 1]: # , 3, 4]:
                        changes.append(
                            Change(self, node1, node2, i)
                        )

        # nb potential new nodes per change
        nb = 0
        for _ in range(nb):
            changes.append(
                Change(self, None, None, 2) # 2 == new-node
            )

        self.changes = changes
        return changes
    
    def copy(self):
        tm = TuringMachine(
            self.batch_size,
            self.nb_inputs,
            self.nb_outputs,
            im_a_copy=True
        )
        # make node copies...
        tm.nodes = [node.copy() for node in self.nodes]

        # copy references
        tm.HALT = tm.nodes[self.HALT.id]

        tm.TRUE = tm.nodes[self.TRUE.id]

        tm.FALSE = tm.nodes[self.FALSE.id]

        tm.outputs = [tm.nodes[id] for id in [n.id for n in self.outputs]]
        tm.inputs = [tm.nodes[id] for id in [n.id for n in self.inputs]]

        return tm
    
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
    
    def get_size(self):
        return len(self.nodes)
    
    def get_nb_nodes(self):
        return self.get_size()
    
    def get_nb_edges(self, old=False):
        # nb = np.sum([n.nb_edges(old) for n in self.nodes])
        nb = self.state.get_nb_edges()

        return nb

    def get_size_cost(self):
        nb_nodes = len(self.nodes)

        # 50 nodes -> 0.1 cost
        # w = 0.1 * np.floor(nb_nodes / 50)
        w = 0.1 / 50 * nb_nodes

        l = w
        return l
    
def binary_diff(x, y):
    right = x == y
    wrong = ~right

    l1 = np.mean(np.abs(wrong))
    return l1

def mse(x, y):
    l = 0.5 * np.abs(x - y) ** 2
    l2 = np.mean(l)

    return l2

def one_hot(x, y):
    """
    classification tasks. assumes x, y are int arrays
    """
    right = x == y
    wrong = ~right

    l1 = np.mean(np.abs(wrong))
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
        xb = post_process_sample(x)

        # xb = encode(x) # not needed after post_process_sample
        xb = np.reshape(xb, xb.shape[0:1] + (-1,))
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

def get_batch():
    global data_loader

    dataset = convert_entire_dataset()
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            )
    
    while True:
        for idx, (xb, yint, yb) in enumerate(data_loader):
            # xb = post_process_sample(x)

            # # xb = encode(x) # not needed after post_process_sample
            # xb = np.reshape(xb, xb.shape[0:1] + (-1,))
            # yb = encode_arr(y)

            # y = y.numpy()
            # yield xb, y, yb
            xb = xb.numpy()
            yint = yint.numpy()
            yb = yb.numpy()
            yield xb, yint, yb




def main():
    # x = D[:, 0:1]
    # y = D[:, 1:2]
    batch_gen = get_batch()

    # xb = np.array([encode(x2) for x2 in x])
    # yb = np.array([encode(y2) for y2 in y])
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
    # old_tm = tm.copy()
    # new_tm = old_tm.copy()
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
            # nb_changes = np.random.randint(0, int(0.1 * len(changes)))

            tm.perturb_x1()
            tm.perturb_x2()
            # ii = np.random.randint(0, len(changes), nb_changes)

            # # get it 
            # for i in ii:
            #     change: Change = changes[i]

            #     # execute it
            #     change.execute()

        # recompile -- not needed since I moved everything to arrays
        # tm.compile()

        # test old a new batch loss. If you want better stats, increase batch_size (rather than putting this section in a for-loop)
        # new batch
        xb, yint, ybgt = next(batch_gen)

        # copy
        # new_tm = tm.copy()

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
            minl = min(avg_l, old_avg_l)
            
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
                # new_tm = old_tm.copy()
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
            print("size: ", tm.get_size())
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
    print("size: ", tm.get_size())
    print("nb edges: ", tm.get_nb_edges())
    # print("gt: ", yb)
    # print("output: ", old_tm.get_output())


    pass


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
else:
    main()