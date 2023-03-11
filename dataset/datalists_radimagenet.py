import os
import yaml
import glob
import math
import random
import pickle
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import collections

def get_file_paths(path):
    return glob.glob(path + "/*")

# 1 for radimagenet v1
# train 80, validation 10, test 10

random.seed(1)


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Lists for containing pseudoname of the patient at its class label
ct_scans_list = []
ct_labels_list = []


# Read 
ct_folders_path = get_file_paths(params_dict.get("radimagenet.directory"))

class_index = 0
for ct_folder_path in ct_folders_path:
    ct_files_path = get_file_paths(ct_folder_path)
    
    for ct_file_path in ct_files_path:
        ct_scans_list.append(ct_file_path)
        ct_labels_list.append(class_index)

    class_index += 1


print(len(ct_scans_list))
print(len(ct_labels_list))

# One hot encoding
"""
ct_labels_array = np.array(ct_labels_list)
ct_labels_array_onehot = np.zeros((ct_labels_array.size, ct_labels_array.max() + 1))
ct_labels_array_onehot[np.arange(ct_labels_array.size), ct_labels_array] = 1

train_ct_scans_list, test_val_ct_scans_list, train_ct_labels_list, test_val_ct_labels_list = train_test_split(ct_scans_list, ct_labels_array_onehot, stratify=ct_labels_array_onehot, test_size=0.2)
val_ct_scans_list, test_ct_scans_list, val_ct_labels_list, test_ct_labels_list = train_test_split(test_val_ct_scans_list, test_val_ct_labels_list, stratify=test_val_ct_labels_list, test_size=0.5)
"""

# Class index
train_ct_scans_list, test_val_ct_scans_list, train_ct_labels_list, test_val_ct_labels_list = train_test_split(ct_scans_list, ct_labels_list, stratify=ct_labels_list, test_size=0.2)
val_ct_scans_list, test_ct_scans_list, val_ct_labels_list, test_ct_labels_list = train_test_split(test_val_ct_scans_list, test_val_ct_labels_list, stratify=test_val_ct_labels_list, test_size=0.5)


#print(train_ct_scans_list[0])
#print(train_ct_labels_list[0])

file_header_name = "radimagenet"

# Train splits
filehandler = open(file_header_name + "_" + str(0.8) + "_" + "train_ct_scans_list.pickle","wb")
pickle.dump(train_ct_scans_list, filehandler)
filehandler.close()

filehandler = open(file_header_name + "_" + str(0.8) + "_" + "train_ct_labels_list.pickle","wb")
pickle.dump(train_ct_labels_list, filehandler)
filehandler.close()


# Validation splits
filehandler = open(file_header_name + "_" + str(0.1) + "_" + "val_ct_scans_list.pickle","wb")
pickle.dump(val_ct_scans_list, filehandler)
filehandler.close()

filehandler = open(file_header_name + "_" + str(0.1) + "_" + "val_ct_labels_list.pickle","wb")
pickle.dump(val_ct_labels_list, filehandler)
filehandler.close()


# Test splits
filehandler = open(file_header_name + "_" + str(0.1) + "_" + "test_ct_scans_list.pickle","wb")
pickle.dump(test_ct_scans_list, filehandler)
filehandler.close()

filehandler = open(file_header_name + "_" + str(0.1) + "_" + "test_ct_labels_list.pickle","wb")
pickle.dump(test_ct_labels_list, filehandler)
filehandler.close()



