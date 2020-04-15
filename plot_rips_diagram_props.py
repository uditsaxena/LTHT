import os
import argparse
import persim
import pickle
import dionysus as dion
import numpy as np
import seaborn as sns
from math import isinf
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)


def count_dgm_props(dgm, dim=0):
    rips_dim = dgm[dim]
    count = 0
    non_inf_list = list(rips_dim[rips_dim[:,1] < np.inf])
    inf_list = list(rips_dim[rips_dim[:,1] == np.inf])
    non_inf_count = len(non_inf_list)
    inf_count = len(inf_list)
    return non_inf_count, inf_count

def calculate_nerual_persistence(dgm, dim):
    rips_dim = dgm[dim]
    non_inf_list = list(rips_dim[rips_dim[:,1] < np.inf])
    dist = np.sum(np.power(np.sum(np.power(non_inf_list, 2), axis=0),1./2.))
    return dist

def compute_rips_diagram_props(all_files):
    rips_dgm_props = defaultdict(list)

    for file in all_files:
        split_name = file.split('/')
        seed, file_name = split_name[-3], split_name[-1]
        # appending '42-8'
        iter_name = "Seed: " + seed + "-" + file_name.split(".")[0]
        print(iter_name)
        rips_file = pickle.load(open(file, 'rb'))
        for dim in [0,1]:
            non_inf_count, inf_count = count_dgm_props(rips_file['dgms'], dim)
            dist = calculate_nerual_persistence(rips_file['dgms'], dim)
            rips_dgm_props[str(dim) + '-non-inf-count'].append(non_inf_count)
            rips_dgm_props[str(dim) + '-inf-count'].append(inf_count)
            rips_dgm_props[str(dim) + '-dist'].append(dist)
        rips_dgm_props['file'].append(iter_name)
        
    return rips_dgm_props

# returns a dataframe
def unpair_for_line_plot(x, y, header):
    file_titles = np.array(list(map(lambda x: x.split('-'), x)))
    values = np.array(y).reshape(-1,1)
    
    values = np.hstack((file_titles, values))
    df = pd.DataFrame(values, columns=['seed','prune_iter',header])
    df['prune_iter'] = df['prune_iter'].astype(float)
    df[header] = df[header].astype(float)
    return df

def plot_dgm_props(dgm_props, directory):
    x=dgm_props['file']
    
    headers = list(dgm_props.keys())
    headers.remove('file')
    for h in headers:
        plt.clf()
        y=dgm_props[h]
        df = unpair_for_line_plot(x, y, h)
        sns.lineplot(x='prune_iter',y=h, hue='seed', data=df, palette=['r','b','g']).set_title(h)
        filename=h
        plt.savefig(directory + filename + ".jpg")

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
    dgm_props = compute_rips_diagram_props(all_files)
    dgm_dir = ROOT_DIR + "{}/{}/".format(model_name, dataset)
    plot_dgm_props(dgm_props, dgm_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/remote_data/saves/", type=str)
    parser.add_argument("--model_name", default='fc1', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
