import sys 
# append the path to folder /train/ in order to use other folders dataset, model...
sys.path.append('/home/guevenira/attention_CT/PDAC/src/')

import yaml
import glob
import pickle
from tqdm import tqdm
import random
import os
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix

from dataset import Dataset
from models.ResNet18.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNet18.ResNet18_MultiheadAttention_v2 import ResNet18_MultiheadAttention_v2
from models.ResNet18.ResNet18_inter_layer_CBAM_MultiheadAttention import ResNet18_inter_layer_MultiheadAttention
from models.ResNet18.ResNet18_maxpool_MultiheadAttention import ResNet18_maxpool_MultiheadAttention
from models.ResNet18.ResNet18_MultiheadAttention_stats_False import ResNet18_MultiheadAttention_stats_False
from models.ResNet18.ResNet18_sum import ResNet18_sum
from models.ResNet18.ResNet18_mean import ResNet18_mean
from models.ResNet18.ResNet18_Linear import ResNet18_Linear
# Transfer learning from RadImageNet
from models.ResNet18.ResNet18_rad_v2_MultiheadAttention import ResNet18_rad_v2_MultiheadAttention


from models.ResNet50.ResNet50_MultiheadAttention import ResNet50_MultiheadAttention
from models.ResNet50.ResNet50_maxpool_MultiheadAttention import ResNet50_maxpool_MultiheadAttention
from models.ResNet50.ResNet50_MultiheadAttention_v2 import ResNet50_MultiheadAttention_v2
from models.ResNet50.ResNet50_sum import ResNet50_sum

from models.ResNet101.ResNet101_MultiheadAttention import ResNet101_MultiheadAttention
from models.ResNet101.ResNet101_sum import ResNet101_sum

from models.ResNet152.ResNet152_MultiheadAttention import ResNet152_MultiheadAttention
from models.ResNet152.ResNet152_MultiheadAttention_v2 import ResNet152_MultiheadAttention_v2
from models.ResNet152.ResNet152_maxpool_MultiheadAttention import ResNet152_maxpool_MultiheadAttention
from models.ResNet152.ResNet152_MultiheadAttention_stats_False import ResNet152_MultiheadAttention_stats_False
from models.ResNet152.ResNet152_inter_layer_CBAM_MultiheadAttention import ResNet152_inter_layer_CBAM_MultiheadAttention

from models.unofficial_ResNet50_CBAM.ResNet50_CBAM_MultiheadAttention_unoffficial import ResNet50_CBAM_MultiheadAttention_unoffficial

"""
from models.ResNetXX_CBAM_MultiheadAttention.ResNet18_CBAM_MultiheadAttention import ResNet18_CBAM_MultiheadAttention
from models.ResNetXX_CBAM_MultiheadAttention.ResNet152_CBAM_MultiheadAttention import ResNet152_CBAM_MultiheadAttention
"""

#from models.ResNetXX_official_CBAM_MultiheadAttention.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNetXX_official_CBAM_MultiheadAttention.ResNet152_CBAM_MultiheadAttention import ResNet152_CBAM_MultiheadAttention

def get_file_paths(path):
    return glob.glob(path + "/*")


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


ct_files_path = get_file_paths(params_dict.get("cts.directory"))
ct_labels_path = params_dict.get("cst.label.csv")
ct_labels_exclude_path = params_dict.get("cts.label.problematic")

ct_labels_df = pd.read_csv(ct_labels_path, index_col=0)
ct_labels_exclude_df = pd.read_csv(ct_labels_exclude_path, index_col=False)


# Getting the current date and time
dt = datetime.now()
ts = int(datetime.timestamp(dt))


torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


NUM_EPOCHS = 100
BATCH_SIZE = 1
lr = 0.001


outputs_folder = "../../results-train/" + params_dict.get("data.label.name").replace("-", "_").lower() + "-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

if params_dict.get("data.label.name") == "Befund-Verlauf":
    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.1_test_ct_scans_list.pickle",'rb')
    test_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_0.1_test_ct_labels_list.pickle",'rb')
    test_ct_labels_list = pickle.load(file)
    file.close()

elif params_dict.get("data.label.name") == "Geschlecht":
    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.1_test_ct_scans_list.pickle",'rb')
    test_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/geschlecht_0.1_test_ct_labels_list.pickle",'rb')
    test_ct_labels_list = pickle.load(file)
    file.close()

elif params_dict.get("data.label.name") == "Befund-Verlauf-Therapie-Procedere":
    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.1_test_ct_scans_list.pickle",'rb')
    test_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/" + params_dict.get("data.label.name.version") + "/befund_verlauf_therapie_procedere_0.1_test_ct_labels_list.pickle",'rb')
    test_ct_labels_list = pickle.load(file)
    file.close()


with open(outputs_folder + "/log.txt", "w") as file:
    print("Class label", params_dict.get("data.label.name"))
    file.write("Class label " + params_dict.get("data.label.name") + "\n")

    print("Class shuffle version", params_dict.get("data.label.name.version"))
    file.write("Class shuffle version " + params_dict.get("data.label.name.version") + "\n")

    print("# of 0s in train: ", train_ct_labels_list.count(0))
    file.write("# of 0s in train: " + str(train_ct_labels_list.count(0)) + "\n")

    print("# of 1s in train: ", train_ct_labels_list.count(1))
    file.write("# of 1s in train: " + str(train_ct_labels_list.count(1)) + "\n")

    print("# of 0s in val: ", val_ct_labels_list.count(0))
    file.write("# of 0s in val: " + str(val_ct_labels_list.count(0)) + "\n")

    print("# of 1s in val: ", val_ct_labels_list.count(1))
    file.write("# of 1s in val: " + str(val_ct_labels_list.count(1)) + "\n")

    print("# of 0s in test: ", test_ct_labels_list.count(0))
    file.write("# of 0s in test: " + str(test_ct_labels_list.count(0)) + "\n")

    print("# of 1s in test: ", test_ct_labels_list.count(1))    
    file.write("# of 1s in test: " + str(test_ct_labels_list.count(1)) + "\n")

    file.close()


if params_dict.get("data.label.balanced"):

    with open(outputs_folder + "/log.txt", "a") as file:
        print("balanced...")
        file.write("balanced..." + "\n")
        file.close()

    # If data is set to be balanced, according to which class label has more instances
    # Shuffle the label that is in excess and train in a ratio of 1:1
    num_of_0s = train_ct_labels_list.count(0)
    num_of_1s = train_ct_labels_list.count(1)

    balanced_count = 0
    if num_of_0s < num_of_1s:
        balanced_count = num_of_0s
    else:
        balanced_count = num_of_1s

    ct_scan_list_balanced = []
    ct_label_list_balanced = []

    counter_0s = 0
    counter_1s = 0

    for i in range(0, len(train_ct_labels_list)):
        if train_ct_labels_list[i] == 0:
            ct_scan_list_balanced.append(train_ct_scans_list[i])
            ct_label_list_balanced.append(train_ct_labels_list[i])
            counter_0s += 1
        if counter_0s == balanced_count:
            break

    for i in range(0, len(train_ct_labels_list)):
        if train_ct_labels_list[i] == 1:
            ct_scan_list_balanced.append(train_ct_scans_list[i])
            ct_label_list_balanced.append(train_ct_labels_list[i])
            counter_1s += 1
        if counter_1s == balanced_count:
            break

    # https://www.geeksforgeeks.org/python-shuffle-two-lists-with-same-order/
    temp_ct_scan_label_list_balanced = list(zip(ct_scan_list_balanced, ct_label_list_balanced))
    random.shuffle(temp_ct_scan_label_list_balanced)
    ct_scan_list_balanced, ct_label_list_balanced = zip(*temp_ct_scan_label_list_balanced)
    train_ct_scans_list, train_ct_labels_list = list(ct_scan_list_balanced), list(ct_label_list_balanced)
    
    with open(outputs_folder + "/log.txt", "a") as file:
        print("# of 0s in train: ", train_ct_labels_list.count(0))
        file.write("# of 0s in train: " + train_ct_labels_list.count(0) + "\n")

        print("# of 1s in train: ", train_ct_labels_list.count(1))
        file.write("# of 1s in train: " + train_ct_labels_list.count(1) + "\n")

        file.close()

# Transforms that need to be applied to the dataset
transforms = {
                'Clip': {'amin': -150, 'amax': 250},

                'Normalize': {'bounds': [-150, 250]},       # Normalize values between 0 and 1

                'Resize': {'height': 256, 'width': 256},    # Original CT layer sizes are 512 x 512

                'Crop-Height' : {'begin': 0, 'end': 256},  # 16 - 240     # 144 - 368
                'Crop-Width' : {'begin': 0, 'end': 256},   # 16 - 240     # 144 - 368

                'limit-max-number-of-layers' : {'bool': True},
                'Max-Layers' : {'max': 200},
                
                'uniform-number-of-layers' : {'bool': False},
                'Uniform-Layers': {'uniform': 110},

                'zero-pad-number-of-layers' : {'bool': False},
                'Zero-Pad-Layers' : {'zeropad': 110},
            }

# FIXME when training only on train instances or train + validation instances
#train_ct_scans_list = train_ct_scans_list + val_ct_scans_list
#train_ct_labels_list = train_ct_labels_list + val_ct_labels_list
# FIXME 

train_dataset = Dataset.Dataset(train_ct_scans_list, train_ct_labels_list, transforms=transforms, train_mode = True)
val_dataset = Dataset.Dataset(val_ct_scans_list, val_ct_labels_list, transforms=transforms, train_mode = False)
test_dataset = Dataset.Dataset(test_ct_scans_list, test_ct_labels_list, transforms=transforms, train_mode = False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18_rad_v2_MultiheadAttention()

# ResNet18_MultiheadAttention
#model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/geschlecht-1677099110/model4.pth'))

# ResNet18_inter_layer_MultiheadAttention after layer 1
#model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/geschlecht-1677199381/model6.pth'))

# ResNet18_inter_layer_MultiheadAttention after layer 1 + layer 2
#model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/geschlecht-1677236627/model9.pth'))

# ResNet_rad_v2 -> this also has the appended conv1 for stepping up channels from 1 to 3
#model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/radimagenet-1677429635/model24.pth'), strict=False)

# ResNet18_rad_v2_MultiheadAttention 
# (1) Transfer learning from RadImageNet from model
# (2) Transfer learning form Geschelcht from trainer
#model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/ResNet18/geschlecht-1677531528/model11.pth'))

model.to(device)


get_params = lambda m: sum(p.numel() for p in m.parameters())
with open(outputs_folder + "/log.txt", "a") as file:
    print(f"Complete model has {get_params(model)} params")
    file.write("Complete model has " + str(get_params(model)) + " params" + "\n")
    file.close()


# loss and optimizer
sigmoid = nn.Sigmoid()
metric = BinaryConfusionMatrix()

# FIXME when training with pos_weight or not
# For ResNetXX_official_CBAM_MultiheadAttention.ResNet152_CBAM_MultiheadAttention use 1.1 for same effect with 1
pos_weight_multiplier = 1
pos_weight = (train_ct_labels_list.count(0) / train_ct_labels_list.count(1)) * pos_weight_multiplier
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
# FIXME 
#criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.01)


train_loss_array = []
train_accuracy_array = []
train_mcc_array = []
train_tp_1_and_2_mcc_array = []
train_tp_0_mcc_array = []

validation_loss_array = []
validation_accuracy_array = []
validation_mcc_array = []

test_loss_array = []
test_accuracy_array = []
test_mcc_array = []
test_tp_1_and_2_mcc_array = []
test_tp_0_mcc_array = []


for epoch in range(NUM_EPOCHS):
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"Epoch: {epoch}")
        file.write("Epoch: " + str(epoch) + "\n")
        file.close()
    
    total_train_loss = 0
    model.train()

    # Comment out for ResNet50 - CBAM MultiheadAttention
    if epoch == 0:
        model.feature_extractor.feature_extractor.eval()
        for param in model.feature_extractor.feature_extractor.parameters():
            param.requires_grad = False
        model.feature_extractor.feature_extractor.fc.weight.requires_grad = True
        model.feature_extractor.feature_extractor.fc.bias.requires_grad = True
    elif epoch == 1:
        model.feature_extractor.feature_extractor.eval()
    elif epoch == 2:
        pass
    elif epoch == 3:
        for param in model.feature_extractor.feature_extractor.layer4.parameters():
            param.requires_grad = True
    elif epoch == 4:
        for param in model.feature_extractor.feature_extractor.layer3.parameters():
            param.requires_grad = True
    elif epoch > 4:
        for param in model.parameters():
            param.requires_grad = True
    

    #### TRAIN ####
    pbar_train_loop = tqdm(train_loader, total=len(train_loader), leave=False)

    train_outputs= []
    train_targets= []
    train_tp_1_and_2_outputs = []
    train_tp_1_and_2_targets = []
    train_tp_0_outputs = []
    train_tp_0_targets = []

    train_iteration = 0
    for train_input, train_target in pbar_train_loop:
        optimizer.zero_grad()

        train_input = train_input.permute(2, 0, 1, 3, 4)
        train_input = torch.squeeze(train_input, 1)

        train_output = model(train_input.to(device))

        train_loss = criterion(train_output, torch.unsqueeze(train_target, 0).float().to(device))
        train_loss.backward()

        optimizer.step()

        train_loss_value = train_loss.detach().cpu().item()
        total_train_loss += train_loss_value
        pbar_train_loop.set_description_str(f"Loss: {train_loss_value:.2f}")

        train_outputs.append(sigmoid(train_output.cpu().flatten()))
        train_targets.append(train_target.flatten())
        
        ct_index = ct_labels_df.index[ct_labels_df['Pseudonym'] == os.path.basename(train_ct_scans_list[train_iteration])[:-4]].tolist()[0]
        if(params_dict.get("data.label.name") == "Befund-Verlauf"):
            if str(ct_labels_df.loc[ct_index]["Therapie-Procedere"]) in ["1", "2"]:
                train_tp_1_and_2_outputs.append(sigmoid(train_output.cpu().flatten()))
                train_tp_1_and_2_targets.append(train_target.flatten())
            elif str(ct_labels_df.loc[ct_index]["Therapie-Procedere"]) in ["0"]:
                train_tp_0_outputs.append(sigmoid(train_output.cpu().flatten()))
                train_tp_0_targets.append(train_target.flatten())
        
        train_iteration += 1

    sched.step()

    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\tMean train loss: {total_train_loss / len(train_loader):.2f}")
        file.write("\tMean train loss: " +  "{:.2f}".format(total_train_loss / len(train_loader)) + "\n")
    
        train_outputs, train_targets = torch.cat(train_outputs), torch.cat(train_targets)
        
        train_accuracy = torchmetrics.functional.accuracy(train_outputs, train_targets, task="binary")
        train_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(train_outputs, train_targets)

        print(f"\tTrain accuracy: {train_accuracy*100.0:.1f}%    Train MCC: {train_mcc*100.0:.1f}%")
        file.write("\tTrain accuracy: " + "{:.1f}".format(train_accuracy*100.0) + "%    Train MCC: " + "{:.1f}".format(train_mcc*100.0) + "%" + "\n")
        
        train_outputs = (train_outputs>0.5).float()
        print('\t' + str(metric(train_outputs, train_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
        file.write('\t' + str(metric(train_outputs, train_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")
        file.close()


    if(params_dict.get("data.label.name") == "Befund-Verlauf"):
        train_tp_1_and_2_outputs, train_tp_1_and_2_targets = torch.cat(train_tp_1_and_2_outputs), torch.cat(train_tp_1_and_2_targets)
        train_tp_1_and_2_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(train_tp_1_and_2_outputs, train_tp_1_and_2_targets)
        train_tp_0_outputs, train_tp_0_targets = torch.cat(train_tp_0_outputs), torch.cat(train_tp_0_targets)
        train_tp_0_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(train_tp_0_outputs, train_tp_0_targets)
        
        with open(outputs_folder + "/log.txt", "a") as file:
            print('\t' + "Train Therapie-Procedere = 1 | 2 ", "MCC: ", "{:.1f}".format(train_tp_1_and_2_mcc*100), "%")
            file.write('\t' + "Train Therapie-Procedere = 1 | 2 " + "MCC: " + "{:.1f}".format(train_tp_1_and_2_mcc*100) + "%" + "\n")

            train_tp_1_and_2_outputs = (train_tp_1_and_2_outputs>0.5).float()
            print('\t' + str(metric(train_tp_1_and_2_outputs, train_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
            file.write('\t' + str(metric(train_tp_1_and_2_outputs, train_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")

            print('\t' + "Train Therapie-Procedere = 0 ", "MCC: ", "{:.1f}".format(train_tp_0_mcc*100), "%")
            file.write('\t' + "Train Therapie-Procedere = 0 " + "MCC: " + "{:.1f}".format(train_tp_0_mcc*100) + "%" + "\n")

            train_tp_0_outputs = (train_tp_0_outputs>0.5).float()
            print('\t' + str(metric(train_tp_0_outputs, train_tp_0_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
            file.write('\t' + str(metric(train_tp_0_outputs, train_tp_0_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")            
            file.close()
        
        train_tp_1_and_2_mcc_array.append(train_tp_1_and_2_mcc)
        train_tp_0_mcc_array.append(train_tp_0_mcc)

    train_loss_array.append(total_train_loss / len(train_loader))
    train_accuracy_array.append(train_accuracy)
    train_mcc_array.append(train_mcc)


    #### VALIDATION ####
    model.eval()

    total_validation_loss = 0
    validation_outputs = []
    validation_targets = []

    with torch.no_grad():
        for validation_input, validation_target in tqdm(val_loader, leave=False):

            validation_input = validation_input.permute(2, 0, 1, 3, 4)
            validation_input = torch.squeeze(validation_input, 1)

            validation_output = model(validation_input.to(device))

            validation_loss = criterion(validation_output, torch.unsqueeze(validation_target, 0).float().to(device))
            validation_loss_value = validation_loss.detach().cpu().item()
            total_validation_loss += validation_loss_value

            validation_outputs.append(sigmoid(validation_output.cpu().flatten()))
            validation_targets.append(validation_target.flatten())
    
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\n\tMean validation loss: {total_validation_loss / len(val_loader):.2f}")
        file.write("\n\tMean validation loss: " +  "{:.2f}".format(total_validation_loss / len(val_loader)) + "\n")
                
        validation_outputs, validation_targets = torch.cat(validation_outputs), torch.cat(validation_targets)
        
        validation_accuracy = torchmetrics.functional.accuracy(validation_outputs, validation_targets, task="binary")
        validation_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(validation_outputs, validation_targets)

        print(f"\tValidation accuracy: {validation_accuracy*100.0:.1f}%    Validation MCC: {validation_mcc*100.0:.1f}%")    
        file.write("\tValidation accuracy: " + "{:.1f}".format(validation_accuracy*100.0) + "%    Validation MCC: " + "{:.1f}".format(validation_mcc*100.0) + "%" + "\n")

        validation_outputs = (validation_outputs>0.5).float()
        print('\t' + str(metric(validation_outputs, validation_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
        file.write('\t' + str(metric(validation_outputs, validation_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")
        file.close()
    
    validation_loss_array.append(total_validation_loss / len(val_loader))
    validation_accuracy_array.append(validation_accuracy)
    validation_mcc_array.append(validation_mcc)


    #### TEST ####
    model.eval()

    total_test_loss = 0
    test_outputs = []
    test_targets = []
    test_tp_1_and_2_outputs = []
    test_tp_1_and_2_targets = []
    test_tp_0_outputs = []
    test_tp_0_targets = []

    test_iteration = 0
    with torch.no_grad():
        for test_input, test_target in tqdm(test_loader, leave=False):

            test_input = test_input.permute(2, 0, 1, 3, 4)
            test_input = torch.squeeze(test_input, 1)

            test_output = model(test_input.to(device))

            test_loss = criterion(test_output, torch.unsqueeze(test_target, 0).float().to(device))
            test_loss_value = test_loss.detach().cpu().item()
            total_test_loss += test_loss_value

            test_outputs.append(sigmoid(test_output.cpu().flatten()))
            test_targets.append(test_target.flatten())

            ct_index = ct_labels_df.index[ct_labels_df['Pseudonym'] == os.path.basename(test_ct_scans_list[test_iteration])[:-4]].tolist()[0]
            if(params_dict.get("data.label.name") == "Befund-Verlauf"):
                if str(ct_labels_df.loc[ct_index]["Therapie-Procedere"]) in ["1", "2"]:
                    test_tp_1_and_2_outputs.append(sigmoid(test_output.cpu().flatten()))
                    test_tp_1_and_2_targets.append(test_target.flatten())
                elif str(ct_labels_df.loc[ct_index]["Therapie-Procedere"]) in ["0"]:
                    test_tp_0_outputs.append(sigmoid(test_output.cpu().flatten()))
                    test_tp_0_targets.append(test_target.flatten())
        
            test_iteration += 1
    
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\n\tMean test loss: {total_test_loss / len(test_loader):.2f}")
        file.write("\n\tMean test loss: " +  "{:.2f}".format(total_test_loss / len(test_loader)) + "\n")
    
        test_outputs, test_targets = torch.cat(test_outputs), torch.cat(test_targets)
        
        test_accuracy = torchmetrics.functional.accuracy(test_outputs, test_targets, task="binary")
        test_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(test_outputs, test_targets)

        print(f"\tTest accuracy: {test_accuracy*100.0:.1f}%    Test MCC: {test_mcc*100.0:.1f}%")
        file.write("\tTest accuracy: " + "{:.1f}".format(test_accuracy*100.0) + "%    Test MCC: " + "{:.1f}".format(test_mcc*100.0) + "%" + "\n")
        
        test_outputs = (test_outputs>0.5).float()
        print('\t' + str(metric(test_outputs, test_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
        file.write('\t' + str(metric(test_outputs, test_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")
        file.close()
    

    if(params_dict.get("data.label.name") == "Befund-Verlauf"):
        test_tp_1_and_2_outputs, test_tp_1_and_2_targets = torch.cat(test_tp_1_and_2_outputs), torch.cat(test_tp_1_and_2_targets)
        test_tp_1_and_2_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(test_tp_1_and_2_outputs, test_tp_1_and_2_targets)
        test_tp_0_outputs, test_tp_0_targets = torch.cat(test_tp_0_outputs), torch.cat(test_tp_0_targets)
        test_tp_0_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(test_tp_0_outputs, test_tp_0_targets)

        with open(outputs_folder + "/log.txt", "a") as file:
            print('\t' + "Test Therapie-Procedere = 1 | 2", "MCC: ", "{:.1f}".format(test_tp_1_and_2_mcc*100), "%")
            file.write('\t' + "Test Therapie-Procedere = 1 | 2 " + "MCC: " + "{:.1f}".format(test_tp_1_and_2_mcc*100) + "%" + "\n")

            test_tp_1_and_2_outputs = (test_tp_1_and_2_outputs>0.5).float()
            print('\t' + str(metric(test_tp_1_and_2_outputs, test_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
            file.write('\t' + str(metric(test_tp_1_and_2_outputs, test_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")

            print('\t' + "Test Therapie-Procedere = 0", "MCC: ", "{:.1f}".format(test_tp_0_mcc*100), "%")
            file.write('\t' + "Test Therapie-Procedere = 0 " + "MCC: " + "{:.1f}".format(test_tp_0_mcc*100) + "%" + "\n")

            test_tp_0_outputs = (test_tp_0_outputs>0.5).float()
            print('\t' + str(metric(test_tp_0_outputs, test_tp_0_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
            file.write('\t' + str(metric(test_tp_0_outputs, test_tp_0_targets).cpu().detach().numpy()).replace('\n', '\n\t') + "\n")            
            file.close()

        test_tp_1_and_2_mcc_array.append(test_tp_1_and_2_mcc)
        test_tp_0_mcc_array.append(test_tp_0_mcc)

    test_loss_array.append(total_test_loss / len(test_loader))
    test_accuracy_array.append(test_accuracy)
    test_mcc_array.append(test_mcc)


    ################################################################################################
    epoch_range = range(epoch + 1)
    epoch_list = [*epoch_range]

    plt.plot(epoch_list, train_loss_array, label="train loss")
    plt.plot(epoch_list, validation_loss_array, label="validation loss")
    plt.plot(epoch_list, test_loss_array, label="test loss")
    plt.legend(loc=4)
    plt.savefig(outputs_folder + "/loss.png")
    plt.close()

    plt.plot(epoch_list, train_accuracy_array, label="train accuracy")
    plt.plot(epoch_list, validation_accuracy_array, label="validation accuracy")
    plt.plot(epoch_list, test_accuracy_array, label="test accuracy")
    plt.legend(loc=4)
    plt.savefig(outputs_folder + "/accuracy.png")
    plt.close()

    plt.plot(epoch_list, train_mcc_array, label="train mcc")
    plt.plot(epoch_list, validation_mcc_array, label="validation mcc")
    plt.plot(epoch_list, test_mcc_array, label="test mcc")
    plt.legend(loc=4)
    plt.savefig(outputs_folder + "/mcc.png")
    plt.close()

    if(params_dict.get("data.label.name") == "Befund-Verlauf"):
        plt.plot(epoch_list, train_tp_1_and_2_mcc_array, label="train tp1&2 mcc")
        plt.plot(epoch_list, train_tp_0_mcc_array, label="train tp0 mcc")
        plt.plot(epoch_list, test_tp_1_and_2_mcc_array, label="test tp1&2 mcc")
        plt.plot(epoch_list, test_tp_0_mcc_array, label="test tp0 mcc")
        plt.legend(loc=4)
        plt.savefig(outputs_folder + "/mcc-tp.png")
        plt.close()

    #### SAVE MODEL WEIGHTS ####
    path_model = outputs_folder + "/model" + str(epoch) + ".pth"
    torch.save(model.state_dict(), path_model)

    with open(outputs_folder + "/log.txt", "a") as file:
        print("\n")
        file.write("\n")
        file.close()