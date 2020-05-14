import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import random
plt.rcParams["figure.figsize"] = (20,20)

def get_dataset_loader(dataset, batch_size):
    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == "mnist":
        testdataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, resnet_nmp

    elif dataset == "cifar10":
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform, download=True)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, resnet_nmp

    elif dataset == "fashionmnist":
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif dataset == "cifar100":
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet
    
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=True)
    
    return test_loader

def compute_accuracy(all_files, test_dataset_loader, restart_at=0):
    accuracy_list = []
    model_list = []
    count = 4
    for listed_file in all_files[restart_at:]:
        best_model_per_pruning_it_location = listed_file + "/" + "model_lt_20.pth.tar"
#         print(best_model_per_pruning_it_location)
        # location looks like: LTHT/remote_data/saves/lenet5_bn/mnist/0/prune_all/global/0/model_lt_20.pth.tar
        split_variables = best_model_per_pruning_it_location.split('/')
        model_name, dataset, seed, epoch = split_variables[-7], split_variables[-6], split_variables[-5], split_variables[-2]
        model_list.append("{}-{} seed:{}-{}".format(model_name, dataset, seed, epoch))
        if (os.path.isfile(best_model_per_pruning_it_location)):
            accuracy = computer_per_model_accuracy(best_model_per_pruning_it_location, test_dataset_loader)
            accuracy_list.append(accuracy)
        #count -=1
        #if (count<=0):
        #    break
    return accuracy_list, model_list



def computer_per_model_accuracy(model_location, test_dataset_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_location, map_location=torch.device(device))
    criterion = nn.CrossEntropyLoss()
    
    accuracy = test(model, test_dataset_loader, criterion)
    print(accuracy)
    return accuracy
#     if dataset == 'mnist':
#         input_dim = (1, 1, 28, 28)
#     elif dataset == 'cifar10':
#         input_dim = (1, 3, 32, 32)

        
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def generate_matrix(accuracy_list):
    original_list = np.asarray(accuracy_list)
    matrix = []
    for score in accuracy_list:
        score_array = np.empty(len(accuracy_list))
        score_array.fill(score)
        score_diff_array = original_list - score_array
        matrix.append(score_diff_array.tolist())
#     print(matrix)
    return matrix

def main(args):
    ROOT_DIR = args.root_dir
    model_name_list = args.model_name
    dataset_list = args.dataset
    seeds = args.seeds
    restart_at = args.restart_at
    prune_all = args.prune_all
    prune_scale = args.prune_scale
    hide_values = args.hide_values
    
    prune_dir_str = ""
    if prune_all:
        prune_dir_str = "prune_all/"
    
    prune_scale_dir_str = ""
    if prune_scale is not None:
        prune_scale_dir_str = str(prune_scale)

    accuracy_list, model_list = [], []
    for model_name in model_name_list:
        all_files = []
        for dataset in dataset_list:
            for seed in seeds:
                model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/{}{}/".format(model_name, dataset, seed, prune_dir_str, prune_scale_dir_str)
                print("In: ", model_dataset_seed_dir)
                files = sorted([model_dataset_seed_dir+f for f in os.listdir(model_dataset_seed_dir) if f.isdigit()])
                all_files.extend(files)

        test_dataset_loader = get_dataset_loader(dataset, args.batch_size)
        per_model_accuracy_list, per_model_model_list = compute_accuracy(all_files, test_dataset_loader, restart_at=restart_at)
        accuracy_list.extend(per_model_accuracy_list)
        model_list.extend(per_model_model_list)
    
    matrix = generate_matrix(accuracy_list)
    prune_str = 'prune_all' if prune_all else ''

    model_name_str = "-".join(model_name_list)
    print(model_name_str)
    dataset_str = "-".join(dataset_list)
    print(dataset_str)
    seeds_str = "-".join(seeds)
    
    filename = ROOT_DIR + "{}-{}-{}_{}".format(model_name_str, dataset_str, seeds_str, prune_str)
    np.save(filename+".npy", matrix)
    heat_map = sns.heatmap(np.asarray(matrix), annot=(not hide_values), xticklabels=model_list, yticklabels=model_list, fmt='.2f', annot_kws={"size": 10}, cmap="YlGnBu")
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    print(filename)
    plt.savefig(filename+".jpg".format(model_name, dataset))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/remote_data/saves/", type=str)
    parser.add_argument("--model_name", default=['lenet5_bn'], nargs='+', type=str)
    parser.add_argument("--dataset", default=['mnist'], nargs='+', type=str)
    parser.add_argument("--seeds", default=['0'], nargs='+', type=str)
    parser.add_argument("--restart_at", default=0, type=int)
    parser.add_argument("--prune_all", action='store_true')
    parser.add_argument("--prune_scale", default='global', type=str)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--hide_values", action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)