import os
import argparse
import persim
import pickle
import dionysus as dion
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


def compute_bottleneck_distance(all_seeds_rips_files, remove_infinity=False):
    matrix = []
    x = []
    y = []
    for file1 in all_seeds_rips_files:
        print('Computing file: {}'.format(file1))
        row = np.zeros(len(all_seeds_rips_files))
        split1_name = file1.split('/')
        seed, file1_name = split1_name[-3], split1_name[-1]
        x.append(seed+"-"+file1_name.split(".")[0])

        rips1 = pickle.load(open(file1, 'rb'))
        if remove_infinity:
            l1 = list(rips1['dgms'][0][rips1['dgms'][0][:,1] < np.inf])
        else:
            l1 = list(rips1['dgms'][0])
        d1 = dion.Diagram(l1)

        for i, file2 in enumerate(all_seeds_rips_files):
            rips2 = pickle.load(open(file2, 'rb'))

            if remove_infinity:
                l2 = list(rips2['dgms'][0][rips2['dgms'][0][:,1] < np.inf])
            else:
                l2 = list(rips2['dgms'][0])

            d2 = dion.Diagram(l2)
            # %time wdist = dion.wasserstein_distance(d1, d2, q=2)
            bdist = dion.bottleneck_distance(d1, d2)
            row[i] = bdist

        matrix.append(row)
    #
    return matrix, x
    #
def main(args):
    ROOT_DIR = args.root_dir
    model_name = args.model_name
    dataset = args.dataset
    seeds = [0, 42, 1337]

    # load list of files
    all_files = []
    for seed in seeds:
        rips_dir = ROOT_DIR + "{}/{}/{}/pickle/".format(model_name, dataset, seed)
        print(rips_dir)
        files = sorted([rips_dir+f for f in os.listdir(rips_dir) if not f.startswith('.')])
        all_files.extend(files)
    matrix, labels = compute_bottleneck_distance(all_files, remove_infinity=args.remove_infinity)
    if args.remove_infinity:
        filename = ROOT_DIR + "{}/{}/".format(model_name, dataset) + "{}-{}_no_inf".format(model_name, dataset)
    else:
        filename = ROOT_DIR + "{}/{}/".format(model_name, dataset) + "{}-{}".format(model_name, dataset)
    np.save(filename+".npy", matrix)
    heat_map = sns.heatmap(np.asarray(matrix), annot=True, xticklabels=labels)
    plt.savefig(filename+".jpg".format(model_name, dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/data/saves/", type=str)
    parser.add_argument("--model_name", default='fc1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--remove_infinity", action='store_true')
#     parser.add_argument("--seed", default='0', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
