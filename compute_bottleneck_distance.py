import os
import argparse
import persim
import pickle

def compute_bottleneck_distance(rips_dir):
    for file1 in os.listdir(rips_dir):
        if file1[0].isdigit():
            prune_iteration1 = file1[0]
            rips1 = pickle.load(open(rips_dir+"/"+file1, 'rb'))
            for file2 in os.listdir(rips_dir):
                if file2[0].isdigit() and file1 != file2:
                    print(file1, file2)
                    prune_iteration2 = file2[0]
                    rips2 = pickle.load(open(rips_dir+"/"+file2, 'rb'))
                    distance_bottleneck, (matching, D) = persim.bottleneck(rips1['dgms'][0], rips2['dgms'][0], matching=True)
                    print('Bottleneck Distance: {}'.format(distance_bottleneck))
                    break
        break
def main(args):
    ROOT_DIR = args.root_dir
    model_name = args.model_name
    dataset = args.dataset
    seed = args.seed

    model_dataset_seed_dir = ROOT_DIR + "{}/{}/{}/pickle".format(model_name, dataset, seed)
    print("In: ", model_dataset_seed_dir)

    if (os.path.isdir(model_dataset_seed_dir)):
        print("computing")
        compute_bottleneck_distance(model_dataset_seed_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="/home/udit/programs/LTHT/remote_data/saves/", type=str)
    parser.add_argument("--model_name", default='lenet5_nmp', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--seed", default='0', type=str)

    args = parser.parse_args()
    print(args)
    main(args)