import os
import argparse
import persim
import pickle
import dionysus as dion
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


def compute_bottleneck_distance(all_seeds_rips_files):
    matrix = []
    x = []
    y = []
    for file1 in all_seeds_rips_files:        
        row = np.zeros(len(all_seeds_rips_files))
        split1_name = file1.split('/')
        seed, file1_name = split1_name[-3], split1_name[-1]
        x.append(seed+"-"+file1_name.split(".")[0])

        rips1 = pickle.load(open(file1, 'rb'))
        d1 = dion.Diagram(list(rips1['dgms'][0]))

        for i, file2 in enumerate(all_seeds_rips_files):
            rips2 = pickle.load(open(file2, 'rb'))
            d2 = dion.Diagram(list(rips2['dgms'][0]))
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
    matrix, labels = compute_bottleneck_distance(all_files)
    filename = ROOT_DIR + "{}/{}/".format(model_name, dataset) + "{}-{}".format(model_name, dataset)
    np.save(filename+".npy", matrix)
    heat_map = sns.heatmap(np.asarray(matrix), annot=True, xticklabels=labels)
    plt.savefig(filename+".jpg".format(model_name, dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/data/saves/", type=str)
    parser.add_argument("--model_name", default='fc1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
#     parser.add_argument("--seed", default='0', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
