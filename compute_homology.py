import networkx as nx
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import pickle
import persim  # see persim.scikit-tda.org
from ripser import ripser  # see ripser.scikit-tda.org

from archs.mnist.fc1 import fc1 as fc1_mnist
from archs.cifar10.fc1 import fc1 as fc1_cifar10

from archs.mnist.AlexNet import AlexNet as AlexNet_mnist
from archs.mnist.AlexNet import AlexNet_nmp as AlexNet_nmp_mnist
from archs.cifar10.AlexNet import AlexNet as AlexNet_cifar10
from archs.cifar10.AlexNet import AlexNet_nmp as AlexNet_nmp_cifar10

from archs.mnist.LeNet5 import LeNet5 as LeNet5_mnist
from archs.mnist.LeNet5 import LeNet5_bn as LeNet5_bn_mnist
from archs.mnist.LeNet5 import LeNet5_nmp as LeNet5_nmp_mnist
from archs.mnist.LeNet5 import LeNet5_nmp_bn as LeNet5_nmp_bn_mnist
from archs.cifar10.LeNet5 import LeNet5 as LeNet5_cifar10
from archs.cifar10.LeNet5 import LeNet5_bn as LeNet5_bn_cifar10
from archs.cifar10.LeNet5 import LeNet5_nmp as LeNet5_nmp_cifar10
from archs.cifar10.LeNet5 import LeNet5_nmp_bn as LeNet5_nmp_bn_cifar10

from archs.mnist.resnet import resnet18 as resnet18_mnist
from archs.mnist.resnet_nmp import resnet18 as resnet18_nmp_mnist
from archs.cifar10.resnet import resnet18 as resnet18_cifar10
from archs.cifar10.resnet_nmp import resnet18 as resnet18_nmp_cifar10

from nn_homology import nn_graph
import argparse

import matplotlib.pyplot as plt

model_graph_dict = {}

def get_model_param_info(model_name, dataset):
    model_param = {
        "fc1_mnist": fc1_mnist,
        "fc1_cifar10": fc1_cifar10,

        "alexnet_mnist": AlexNet_mnist,
        "alexnet_nmp_mnist": AlexNet_nmp_mnist,
        "alexnet_cifar10": AlexNet_cifar10,
        "alexnet_nmp_cifar10": AlexNet_nmp_cifar10,

        "lenet5_mnist": LeNet5_mnist,
        "lenet5_bn_mnist": LeNet5_bn_mnist,
        "lenet5_nmp_mnist": LeNet5_nmp_mnist,
        "lenet5_nmp_bn_mnist": LeNet5_nmp_bn_mnist,
        "lenet5_cifar10": LeNet5_cifar10,
        "lenet5_bn_cifar10": LeNet5_bn_cifar10,
        "lenet5_nmp_cifar10": LeNet5_nmp_cifar10,
        "lenet5_nmp_bn_cifar10": LeNet5_nmp_bn_cifar10,

        "resnet18_mnist": resnet18_mnist,
        "resnet18_nmp_mnist": resnet18_nmp_mnist,
        "resnet18_cifar10": resnet18_cifar10,
        "resnet18_nmp_cifar10": resnet18_nmp_cifar10
    }
    architecture = model_name + "_" + dataset
    print("Getting parameters for: ", architecture)
    param_info = model_param[architecture]().param_info
    return param_info


def compute_homology(model, dataset, root_dir, restart_at=0):

    init_model_location = root_dir + "initial_state_dict_lt.pth.tar"
    print('Computing Homology of Initial Parameters')
    computer_per_model_homology(model, dataset, root_dir, 'init',
                                init_model_location)

    for listed_file in sorted(os.listdir(root_dir))[restart_at:]:
        if (listed_file[0].isdigit()):
            print("epoch: ", listed_file)
            if listed_file != ".ipynb_checkpoints":
                best_model_per_pruning_it_location = root_dir + listed_file + "/" + "model_lt_20.pth.tar"
                # print(best_model_per_pruning_it_location)
                if (os.path.isfile(best_model_per_pruning_it_location)):
                    computer_per_model_homology(model, dataset, root_dir, listed_file,
                                                best_model_per_pruning_it_location)


def sparse_min_row(csr_mat):
    ret = np.zeros(csr_mat.shape[0])
    ret[np.diff(csr_mat.indptr) != 0] = np.minimum.reduceat(csr_mat.data,csr_mat.indptr[:-1][np.diff(csr_mat.indptr)>0])
    return ret

def computer_per_model_homology(model_name, dataset, root_dir, epoch, model_location):
    rips_pickle_dir = root_dir + "pickle/"
    # print(rips_pickle_dir)
    persim_image_dir = root_dir + "persim/"
    # print(persim_image_dir)

    model = torch.load(model_location, map_location=torch.device('cpu'))
    if dataset == 'mnist':
        input_dim = (1, 1, 28, 28)
    elif dataset == 'cifar10':
        input_dim = (1, 3, 32, 32)

    param_info = get_model_param_info(model_name, dataset)

    architecture = model_name + "_" + dataset
    if (architecture not in model_graph_dict) or (epoch == 0):
        print(("Architecture: {} not found, creating").format(architecture))
        NNG = nn_graph.NNGraph()
        NNG.parameter_graph(model, param_info, input_dim, ignore_zeros=True, verbose=True)
        # model_graph_dict[architecture] = NNG
    else:
        print(("Architecture: {} found, loading ... ").format(architecture))
        # NNG = model_graph_dict[architecture]
        # NNG.update_adjacency(model)

    print('Computing Homology')
    sps = nx.to_scipy_sparse_matrix(NNG.G)
    mrs = sparse_min_row(sps)
    sps.setdiag(mrs)
    rips = ripser(sps, distance_matrix=True, maxdim=1, do_cocycles=True)

    # root_dir contains something in the format of:
    # /home/udit/programs/LTHT/remote_data/saves/alexnet_nmp/mnist/0/

    if not (os.path.isdir(rips_pickle_dir)):
        os.mkdir(rips_pickle_dir)
    rips_file = rips_pickle_dir + epoch
    rips_pickle = open(rips_file + ".pickle", "wb")
    pickle.dump(rips, rips_pickle)
    rips_pickle.close()

    # save ripser file as pickle
    persim.plot_diagrams(rips['dgms'])

    if not (os.path.isdir(persim_image_dir)):
        os.mkdir(persim_image_dir)
    persim_plot_file = persim_image_dir + epoch
    plt.savefig(persim_plot_file + ".jpg")
    plt.clf()


def main(args):
    ROOT_DIR = args.root_dir
    model_name = args.model_name
    dataset = args.dataset
    seed = args.seed
    restart_at = args.restart_at
    prune_all = args.prune_all
    prune_scale = args.prune_scale

    if prune_scale is not None:
        if prune_all:
            model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/prune_all/{}/".format(model_name, dataset, seed, prune_scale)
        else:
            model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/{}/".format(model_name, dataset, seed, prune_scale)
    else:
        if prune_all:
            model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/prune_all/".format(model_name, dataset, seed)
        else:
            model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/".format(model_name, dataset, seed)
    print("In: ", model_dataset_seed_dir)

    if (os.path.isdir(model_dataset_seed_dir)):
        compute_homology(model_name, dataset, model_dataset_seed_dir, restart_at=restart_at)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/remote_data/saves/", type=str)
    parser.add_argument("--model_name", default='lenet5_nmp', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--seed", default='0', type=str)
    parser.add_argument("--restart_at", default=0, type=int)
    parser.add_argument("--prune_all", action='store_true')
    parser.add_argument("--prune_scale", default=None, type=str)

    args = parser.parse_args()
    print(args)
    main(args)
