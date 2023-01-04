import sys 
# append the path to folder /train/ in order to use other folders dataset, model...
sys.path.append('/home/guevenira/attention_CT/PDAC/src/')

import yaml
import glob
import pickle

from dataset import dataset


def get_file_paths(path):
    return glob.glob(path + "/*")


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Transforms that need to be applied to the dataset
transforms = {
                'Clip': {'amin': -150, 'amax': 250},

                'Normalize': {'bounds': [-150, 250]},       # Normalize values between 0 and 1

                'Resize': {'height': 256, 'width': 256},    # Original CT layer sizes are 512 x 512

                'Crop-Height' : {'begin': 0, 'end': 256},
                'Crop-Width' : {'begin': 0, 'end': 256},

                'limit-max-number-of-layers' : {'bool': True},
                'Max-Layers' : {'max': 195},
                
                'uniform-number-of-layers' : {'bool': False},
                'Uniform-Layers': {'uniform': 200},
            }


NUM_EPOCHS = 20
BATCH_SIZE = 1
lr = 0.001


if params_dict.get("data.label.name") == "Befund-Verlauf":
    file = open("../dataset/befund_verlauf_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

elif params_dict.get("data.label.name") == "Geschlecht":
    file = open("../dataset/geschlecht_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()



print(train_ct_labels_list.count(0))
print(train_ct_labels_list.count(1))

print(val_ct_labels_list.count(0))
print(val_ct_labels_list.count(1))

exit()



train_dataset = dataset.Dataset(ct_scan_list[: train_size], ct_label_list[: train_size], transforms=transforms)
val_dataset = dataset.Dataset(ct_scan_list[train_size : train_size + val_size], ct_label_list[train_size : train_size + val_size], transforms=transforms)
test_dataset = dataset.Dataset(ct_scan_list[train_size + val_size :], ct_label_list[train_size + val_size :], transforms=transforms)
