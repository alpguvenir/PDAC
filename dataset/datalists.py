import os
import yaml
import glob
import math

import pandas as pd

def get_file_paths(path):
    return glob.glob(path + "/*")


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
            if(params_dict.get("data.label.name") == "Geschlecht"):
                if not(math.isnan(ct_labels_df.loc[ct_index]['Geschlecht'].item())):
                    ct_scans_list.append(ct_file_path)
                    # Labels are 0 or 1
                    ct_labels_list.append(int(ct_labels_df.loc[ct_index]['Geschlecht']))

            # Befund Verlauf
            elif(params_dict.get("data.label.name") == "Befund-Verlauf"):
                if str(ct_labels_df.loc[ct_index]["Befund Verlauf"]) in ['SD', 'PR', 'RD', 'SD-RD']:
                    ct_scans_list.append(ct_file_path)
                    ct_labels_list.append(0)
                elif str(ct_labels_df.loc[ct_index]["Befund Verlauf"]) in ['PD', 'PD (MRT keine Leberläsionen)']:
                    ct_scans_list.append(ct_file_path)
                    ct_labels_list.append(1)

