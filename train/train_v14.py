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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix

from dataset import Dataset
from models.ResNet18.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNet18.ResNet18_MultiheadAttention_stats_False import ResNet18_MultiheadAttention_stats_False
from models.ResNet18.ResNet18_sum import ResNet18_sum
from models.ResNet18.ResNet18_mean import ResNet18_mean
from models.ResNet18.ResNet18_Linear import ResNet18_Linear

from models.ResNet50.ResNet50_MultiheadAttention import ResNet50_MultiheadAttention
from models.ResNet50.ResNet50_sum import ResNet50_sum

from models.ResNet101.ResNet101_MultiheadAttention import ResNet101_MultiheadAttention
from models.ResNet101.ResNet101_sum import ResNet101_sum

from models.ResNet152.ResNet152_MultiheadAttention import ResNet152_MultiheadAttention
from models.ResNet152.ResNet152_MultiheadAttention_stats_False import ResNet152_MultiheadAttention_stats_False
from models.ResNet152.ResNet152_CBAM_MultiheadAttention import ResNet152_CBAM_MultiheadAttention

from models.unofficial_ResNet50_CBAM.ResNet50_CBAM_MultiheadAttention_unoffficial import ResNet50_CBAM_MultiheadAttention_unoffficial

from models.VGG16.VGG16_MultiheadAttention import VGG16_MultiheadAttention
from models.VGG19.VGG19_MultiheadAttention import VGG19_MultiheadAttention

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


NUM_EPOCHS = 50
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


print("Class label", params_dict.get("data.label.name"))
print("Class shuffle version", params_dict.get("data.label.name.version"))

print("# of 0s in train: ", train_ct_labels_list.count(0))
print("# of 1s in train: ", train_ct_labels_list.count(1))

print("# of 0s in val: ", val_ct_labels_list.count(0))
print("# of 1s in val: ", val_ct_labels_list.count(1))

print("# of 0s in test: ", test_ct_labels_list.count(0))
print("# of 1s in test: ", test_ct_labels_list.count(1))

if params_dict.get("data.label.balanced"):

    print("balanced...")
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
    
    print("# of 0s in train: ", train_ct_labels_list.count(0))
    print("# of 1s in train: ", train_ct_labels_list.count(1))

# Transforms that need to be applied to the dataset
transforms = {
                'Clip': {'amin': -150, 'amax': 250},

                'Normalize': {'bounds': [-150, 250]},       # Normalize values between 0 and 1

                'Resize': {'height': 256, 'width': 256},    # Original CT layer sizes are 512 x 512

                'Crop-Height' : {'begin': 0, 'end': 256},
                'Crop-Width' : {'begin': 0, 'end': 256},

                'limit-max-number-of-layers' : {'bool': True},
                'Max-Layers' : {'max': 200},
                
                'uniform-number-of-layers' : {'bool': False},
                'Uniform-Layers': {'uniform': 200},

                'zero-pad-number-of-layers' : {'bool': False},
                'Zero-Pad-Layers' : {'zeropad': 100},
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
# ResNet50_MultiheadAttention
# ResNet50
model = VGG19_MultiheadAttention()
model.to(device)


get_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"Complete model has {get_params(model)} params")


# loss and optimizer
sigmoid = nn.Sigmoid()
metric = BinaryConfusionMatrix()

# FIXME when training with pos_weight or not
#pos_weight_multiplier = 0.7
#pos_weight = (train_ct_labels_list.count(0) / train_ct_labels_list.count(1)) * pos_weight_multiplier
#criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
# FIXME 
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.01)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    total_train_loss = 0
    model.train()

    if epoch == 0:
        model.feature_extractor.eval()
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        model.feature_extractor.features[0].weight.requires_grad = True
        model.feature_extractor.features[0].bias.requires_grad = True
    
    
    #### TRAIN ####
    pbar_train_loop = tqdm(train_loader, total=len(train_loader), leave=False)

    train_outputs= []
    train_targets= []
    train_tp_1_and_2_outputs = []
    train_tp_1_and_2_targets = []

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
        
        train_iteration += 1

    sched.step()

    print(f"\tMean train loss: {total_train_loss / len(train_loader):.2f}")
    train_outputs, train_targets = torch.cat(train_outputs), torch.cat(train_targets)
    
    train_accuracy = torchmetrics.functional.accuracy(train_outputs, train_targets, task="binary")
    train_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(train_outputs, train_targets)

    print(f"\tTrain accuracy: {train_accuracy*100.0:.1f}%")
    print(f"\tTrain MCC: {train_mcc*100.0:.1f}%")

    train_outputs = (train_outputs>0.5).float()
    print('\t' + str(metric(train_outputs, train_targets).cpu().detach().numpy()).replace('\n', '\n\t'))

    
    print('\t' + "Train Therapie-Procedere")
    train_tp_1_and_2_outputs, train_tp_1_and_2_targets = torch.cat(train_tp_1_and_2_outputs), torch.cat(train_tp_1_and_2_targets)
    train_tp_1_and_2_outputs = (train_tp_1_and_2_outputs>0.5).float()
    print('\t' + str(metric(train_tp_1_and_2_outputs, train_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
    

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
    
    print(f"\tMean validation loss: {total_validation_loss / len(val_loader):.2f}")
    validation_outputs, validation_targets = torch.cat(validation_outputs), torch.cat(validation_targets)
    
    validation_accuracy = torchmetrics.functional.accuracy(validation_outputs, validation_targets, task="binary")
    validation_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(validation_outputs, validation_targets)

    print(f"\tValidation accuracy: {validation_accuracy*100.0:.1f}%")
    print(f"\tValidation MCC: {validation_mcc*100.0:.1f}%")

    validation_outputs = (validation_outputs>0.5).float()
    print('\t' + str(metric(validation_outputs, validation_targets).cpu().detach().numpy()).replace('\n', '\n\t'))


    #### TEST ####
    model.eval()

    total_test_loss = 0
    test_outputs = []
    test_targets = []
    test_tp_1_and_2_outputs = []
    test_tp_1_and_2_targets = []

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
        
            test_iteration += 1
    
    print(f"\tMean test loss: {total_test_loss / len(test_loader):.2f}")
    test_outputs, test_targets = torch.cat(test_outputs), torch.cat(test_targets)
    
    test_accuracy = torchmetrics.functional.accuracy(test_outputs, test_targets, task="binary")
    test_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(test_outputs, test_targets)

    print(f"\tTest accuracy: {test_accuracy*100.0:.1f}%")
    print(f"\tTest MCC: {test_mcc*100.0:.1f}%")

    test_outputs = (test_outputs>0.5).float()
    print('\t' + str(metric(test_outputs, test_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
    
    
    print('\t' + "Test Therapie-Procedere")
    test_tp_1_and_2_outputs, test_tp_1_and_2_targets = torch.cat(test_tp_1_and_2_outputs), torch.cat(test_tp_1_and_2_targets)
    test_tp_1_and_2_outputs = (test_tp_1_and_2_outputs>0.5).float()
    print('\t' + str(metric(test_tp_1_and_2_outputs, test_tp_1_and_2_targets).cpu().detach().numpy()).replace('\n', '\n\t'))
    

    #### SAVE MODEL WEIGHTS ####
    path_model = outputs_folder + "/model" + str(epoch) + ".pth"
    torch.save(model.state_dict(), path_model)

    print("\n")