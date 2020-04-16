import networkx as nx
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import scipy.sparse

import pandas as pd
from archs.mnist.fc1 import fc1 as fc1_mnist
from archs.cifar10.fc1 import fc1 as fc1_cifar10

from archs.mnist.AlexNet import AlexNet as AlexNet_mnist
from archs.mnist.AlexNet import AlexNet_nmp as AlexNet_nmp_mnist
from archs.cifar10.AlexNet import AlexNet as AlexNet_cifar10
from archs.cifar10.AlexNet import AlexNet_nmp as AlexNet_nmp_cifar10

from archs.mnist.LeNet5 import LeNet5 as LeNet5_mnist
from archs.mnist.LeNet5 import LeNet5_nmp as LeNet5_nmp_mnist
from archs.cifar10.LeNet5 import LeNet5 as LeNet5_cifar10
from archs.cifar10.LeNet5 import LeNet5_nmp as LeNet5_nmp_cifar10

from archs.mnist.resnet import resnet18 as resnet18_mnist
from archs.mnist.resnet_nmp import resnet18 as resnet18_nmp_mnist
from archs.cifar10.resnet import resnet18 as resnet18_cifar10
from archs.cifar10.resnet_nmp import resnet18 as resnet18_nmp_cifar10

from nn_homology import nn_graph
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

model_graph_dict = {}

def get_model_param_info(model_name, dataset):
    model_param = {
        "fc1_mnist": fc1_mnist,
        "fc1_cifar10": fc1_cifar10,

        "alexnet_mnist": AlexNet_mnist,
        "alexnet_nmp_mnist": AlexNet_nmp_mnist,
        "alexnet_cifar10": AlexNet_cifar10,
        "alexnet_nmp_cifar10": AlexNet_nmp_cifar10,

        "lenet5_mnist": AlexNet_mnist,
        "lenet5_nmp_mnist": LeNet5_nmp_mnist,
        "lenet5_cifar10": LeNet5_cifar10,
        "lenet5_nmp_cifar10": LeNet5_nmp_cifar10,

        "resnet18_mnist": resnet18_mnist,
        "resnet18_nmp_mnist": resnet18_nmp_mnist,
        "resnet18_cifar10": resnet18_cifar10,
        "resnet18_nmp_cifar10": resnet18_nmp_cifar10
    }
    architecture = model_name + "_" + dataset
    print("Getting parameters for: ", architecture)
    param_info = model_param[architecture]().param_info
    return param_info


def compute_graph_metrics(model_name, dataset, seed, root_dir):
    columns = ["prune_iter", "s_metric", "wiener_index", "avg_clustering", "node_connectivity", "diameter", "local_efficiency", "global_efficiency", "overall_reciprocity"]
    rows = []
    for prune_iter in sorted(os.listdir(root_dir)):
        if (prune_iter[0].isdigit()):
            print("prune_iter: ", prune_iter)
            if prune_iter != ".ipynb_checkpoints":
                best_model_per_pruning_it_location = root_dir + prune_iter + "/" + "model_lt_20.pth.tar"
                # print(best_model_per_pruning_it_location)
                if (os.path.isfile(best_model_per_pruning_it_location)):
                    row = [prune_iter]
                    metrics_list = compute_model_graph_metrics(model_name, dataset, root_dir, prune_iter,
                                               best_model_per_pruning_it_location)
                    row.append(metrics_list)
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    for metric in columns: 
        plt.clf()

        sns.lineplot(x='prune_iter',y=metric, data=df).set_title("prune_iter vs {}".format(metric))
        file_loc = root_dir + "{}-{}-{}-{}".format(model_name, dataset, seed, metric)
        plt.savefig(file_loc+".jpg")

def compute_model_graph_metrics(model_name, dataset, root_dir, epoch, model_location):
    model = torch.load(model_location)
    if dataset == 'mnist':
        input_dim = (1, 1, 28, 28)
    elif dataset == 'cifar10':
        input_dim = (1, 3, 32, 32)

    param_info = get_model_param_info(model_name, dataset)

    architecture = model_name + "_" + dataset
    if (architecture not in model_graph_dict) or (epoch == 0):
        print(("Architecture: {} not found, creating").format(architecture))
        NNG = nn_graph.NNGraph()
        NNG.parameter_graph(model, param_info, input_dim, ignore_zeros=True)
        
    s_metric = nx.s_metric(NNG.G, normalized=False)
    wiener_index = nx.wiener_index(NNG.G, weight='weight')
    avg_clustering = nx.average_clustering(NNG.G, weight='weight')
    node_connectivity = nx.node_connectivity(NNG.G)
    
    diameter = nx.diameter(NNG.G)
    local_efficiency = nx.local_efficiency(NNG.G)
    global_efficiency = nx.global_efficiency(NNG.G)
    overall_reciprocity = nx.overall_reciprocity(NNG.G)
    
    return [s_metric, wiener_index, avg_clustering, node_connectivity, diameter, local_efficiency, global_efficiency, overall_reciprocity]
    
def main(args):
    ROOT_DIR = args.root_dir
    model_name_list = args.model_name
    dataset_list = args.dataset
    seed_list = args.seed
    
    for model_name in model_name_list:
        for dataset in dataset_list:
            for seed in seed_list:
                model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/".format(model_name, dataset, seed)
                print("In: ", model_dataset_seed_dir)

                if (os.path.isdir(model_dataset_seed_dir)):
                    compute_graph_metrics(model_name, dataset, seed, model_dataset_seed_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/remote_data/saves/", type=str)
    parser.add_argument("--model_name", nargs='+', default='fc1', type=str)
    parser.add_argument("--dataset", nargs='+', default='mnist', type=str)
    parser.add_argument("--seed", nargs='+', default='0', type=str)

    args = parser.parse_args()
    print(args)
    main(args)