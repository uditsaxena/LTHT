import os
import argparse
import persim
import pickle
import dionysus as dion
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


def compute_bottleneck_distance(all_seeds_rips_files, remove_infinity=False, compute_wass_distance=False,
                                use_persim=False, M=10):
    matrix = []
    x = []
    y = []
    for file1 in all_seeds_rips_files:
        print('Computing file: {}'.format(file1))
        row = np.zeros(len(all_seeds_rips_files))
        # example file1: LTHT/remote_data/saves/alexnet_nmp/mnist/42/pickle/8.pickle
        split1_name = file1.split('/')
        seed, file1_name = split1_name[-4], split1_name[-1]
        # appending '42-8'
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

            if compute_wass_distance:
                if use_persim:
                    wdist = persim.sliced_wasserstein_kernel(d1,d2,M=M)
                else:
                    wdist = dion.wasserstein_distance(d1, d2, q=2)
                row[i] = wdist
            else:
                if use_persim:
                    bdist = persim.bottleneck(d1,d2)
                else:
                    bdist = dion.bottleneck_distance(d1, d2)
                row[i] = bdist

        matrix.append(row)
    #
    x = list(map(lambda y:'Seed {}-{}'.format(y.split('-')[0], y.split('-')[1]), x))
    return matrix, x
    #
def main(args):
    ROOT_DIR = args.root_dir
    model_name_list = args.model_name
    dataset_list = args.dataset
    seeds = [0, 42, 1337]
    wass = args.compute_wass_distance
    persim = args.use_persim
    M = args.M
    prune_all = args.prune_all

    dist_type = 'wasserstein' if wass else 'bottleneck'

    # load list of files
    all_files = []
    for model_name in model_name_list:
        for dataset in dataset_list:
            for seed in seeds:
                if prune_all:
                    rips_dir = ROOT_DIR + "{}/{}/{}/prune_all/pickle/".format(model_name, dataset, seed)
                else:
                    rips_dir = ROOT_DIR + "{}/{}/{}/pickle/".format(model_name, dataset, seed)
                print(rips_dir)
                files = sorted([rips_dir+f for f in os.listdir(rips_dir) if not f.startswith('.')])
                all_files.extend(files)
    print(all_files)
    matrix, labels = compute_bottleneck_distance(all_files,
                                                 remove_infinity=args.remove_infinity,
                                                 compute_wass_distance=args.compute_wass_distance,
                                                 M=M, use_persim=persim)

    prune_str = 'prune_all' if prune_all else ''
    if args.remove_infinity:
        filename = ROOT_DIR + "{}/{}/".format(model_name, dataset) + "{}-{}_{}_{}_no_inf".format(model_name, dataset, dist_type, prune_str)
    else:
        filename = ROOT_DIR + "{}/{}/".format(model_name, dataset) + "{}-{}_{}_{}".format(model_name, dataset, dist_type, prune_str)
    np.save(filename+".npy", matrix)
    heat_map = sns.heatmap(np.asarray(matrix), annot=True, xticklabels=labels, yticklabels=labels, fmt='.2f', annot_kws={"size": 10})
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    plt.savefig(filename+".jpg".format(model_name, dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/data/saves/", type=str)
    parser.add_argument("--model_name", default='fc1', nargs='+', type=str)
    parser.add_argument("--dataset", default='mnist', nargs='+', type=str)
    parser.add_argument("--remove_infinity", action='store_true')
    parser.add_argument("--use-persim", action='store_true')
    parser.add_argument("--M", default=10, type=int)
    parser.add_argument("--compute_wass_distance", action='store_true',
                        help="Compute wasserstein distance instead of bottleneck distance")
    parser.add_argument("--prune_all", action='store_true')


    args = parser.parse_args()
    print(args)
    main(args)
