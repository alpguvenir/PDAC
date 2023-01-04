import os
import yaml
import glob
import math
import random
import pickle
from sklearn.model_selection import train_test_split

import pandas as pd


def get_file_paths(path):
    return glob.glob(path + "/*")


random.seed(1)


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Lists for containing pseudoname of the patient at its class label
ct_scans_list = []
ct_labels_list = []


# Read 
ct_files_path = get_file_paths(params_dict.get("cts.directory"))
ct_labels_path = params_dict.get("cst.label.csv")
ct_labels_exclude_path = params_dict.get("cts.label.problematic")

ct_labels_df = pd.read_csv(ct_labels_path, index_col=0)
ct_labels_exclude_df = pd.read_csv(ct_labels_exclude_path, index_col=False)


# For each CT instance
for ct_file_path in ct_files_path:
    ct_file_name = os.path.basename(ct_file_path)

    # Check if patient name exists only once in pseudonymised_patient_info.csv
    if len(ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()) == 1:
        ct_index = ct_labels_df.index[ct_labels_df['Pseudonym'] == ct_file_name[:-4]].tolist()[0]

        # Check if the patient name is not in Problematic_CTs
        if len(ct_labels_exclude_df.index[ct_labels_exclude_df['Patient_name'] == ct_file_name[:-4]].tolist()) == 0:
    
            # Geschlecht
            # 412 - 0 instances
            # 453 - 1 instances
            if(params_dict.get("data.label.name") == "Geschlecht"):
                if not(math.isnan(ct_labels_df.loc[ct_index]['Geschlecht'].item())):
                    ct_scans_list.append(ct_file_path)
                    # Labels are 0 or 1
                    ct_labels_list.append(int(ct_labels_df.loc[ct_index]['Geschlecht']))

            # Befund Verlauf
            # 352 - 0 instances
            # 186 - 1 instances
            elif(params_dict.get("data.label.name") == "Befund-Verlauf"):
                if str(ct_labels_df.loc[ct_index]["Befund Verlauf"]) in ['SD', 'PR', 'RD', 'SD-RD']:
                    ct_scans_list.append(ct_file_path)
                    ct_labels_list.append(0)
                elif str(ct_labels_df.loc[ct_index]["Befund Verlauf"]) in ['PD', 'PD (MRT keine Leberläsionen)']:
                    ct_scans_list.append(ct_file_path)
                    ct_labels_list.append(1)


temp_ct_scans_labels_list = list(zip(ct_scans_list, ct_labels_list))
random.shuffle(temp_ct_scans_labels_list)


ct_scans_list, ct_labels_list = zip(*temp_ct_scans_labels_list)
ct_scans_list, ct_labels_list = list(ct_scans_list), list(ct_labels_list)

# num_of_0s = ct_labels_list.count(0)
# num_of_1s = ct_labels_list.count(1)


train_ct_scans_list, test_val_ct_scans_list, train_ct_labels_list, test_val_ct_labels_list = train_test_split(ct_scans_list, ct_labels_list, stratify=ct_labels_list, test_size=0.2)
val_ct_scans_list, test_ct_scans_list, val_ct_labels_list, test_ct_labels_list = train_test_split(test_val_ct_scans_list, test_val_ct_labels_list, stratify=test_val_ct_labels_list, test_size=0.5)


file_header_name = params_dict.get("data.label.name").replace("-", "_").lower()


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


print(train_ct_labels_list.count(0))
print(val_ct_labels_list.count(0))
print(test_ct_labels_list.count(0))

print(train_ct_labels_list.count(1))
print(val_ct_labels_list.count(1))
print(test_ct_labels_list.count(1))