import sys 
# append the path to folder /train/ in order to use other folders dataset, model...
sys.path.append('/home/guevenira/attention_CT/PDAC/src/')

# import dependencies
import yaml
import glob
import os
import math
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import random
import pickle

# import pytorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import for datalader
from tqdm import tqdm

# import models and dataloader
from dataset import Dataset
from models.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNet18_mean import ResNet18_mean
from models.ResNet18_sum import ResNet18_sum
from models.ResNet18_Linear import ResNet18_Linear

import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


def get_file_paths(path):
    return glob.glob(path + "/*")


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Getting the current date and time
dt = datetime.now()
ts = int(datetime.timestamp(dt))

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

BATCH_SIZE = 1

outputs_folder = "../../results-test/test-v1-" + params_dict.get("data.label.name").replace("-", "_").lower() + "-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

if params_dict.get("data.label.name") == "Befund-Verlauf":
    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.1_test_ct_scans_list.pickle",'rb')
    test_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/befund_verlauf_v1/befund_verlauf_0.1_test_ct_labels_list.pickle",'rb')
    test_ct_labels_list = pickle.load(file)
    file.close()

elif params_dict.get("data.label.name") == "Geschlecht":
    file = open("../dataset/geschlecht_v1/geschlecht_0.8_train_ct_scans_list.pickle",'rb')
    train_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_v1/geschlecht_0.8_train_ct_labels_list.pickle",'rb')
    train_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_v1/geschlecht_0.1_val_ct_scans_list.pickle",'rb')
    val_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_v1/geschlecht_0.1_val_ct_labels_list.pickle",'rb')
    val_ct_labels_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_v1/geschlecht_0.1_test_ct_scans_list.pickle",'rb')
    test_ct_scans_list = pickle.load(file)
    file.close()

    file = open("../dataset/geschlecht_v1/geschlecht_0.1_test_ct_labels_list.pickle",'rb')
    test_ct_labels_list = pickle.load(file)
    file.close()


print("# of 0s in train: ", train_ct_labels_list.count(0))
print("# of 1s in train: ", train_ct_labels_list.count(1))

print("# of 0s in val: ", val_ct_labels_list.count(0))
print("# of 1s in val: ", val_ct_labels_list.count(1))

print("# of 0s in test: ", test_ct_labels_list.count(0))
print("# of 1s in test: ", test_ct_labels_list.count(1))


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
            }




test_dataset = Dataset.Dataset(test_ct_scans_list, test_ct_labels_list, transforms=transforms, train_mode = False)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# Use gpu if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = ResNet18_MultiheadAttention()
model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/geschlecht-1673804012/model6.pth'))
model.to(device)


# loss and optimizer
sigmoid = nn.Sigmoid()
criterion = nn.BCEWithLogitsLoss()


#### TEST ####
model.eval()

total_test_loss = 0
test_outputs = []
test_targets = []

with torch.no_grad():
    for test_input, test_target in tqdm(test_loader, leave=False):

        test_input = test_input.permute(2, 0, 1, 3, 4)
        test_input = torch.squeeze(test_input, 1)

        test_output, att_map = model(test_input.to(device))

        print(att_map.shape)
        print(torch.max(att_map))
        print(torch.min(att_map))

        test_loss = criterion(test_output, torch.unsqueeze(test_target, 0).float().to(device))

        test_loss_value = test_loss.detach().cpu().item()

        total_test_loss += test_loss_value

        test_outputs.append(sigmoid(test_output.cpu().flatten()))
        test_targets.append(test_target.flatten())


print(f"\tMean test loss: {total_test_loss / len(test_loader):.2f}")
test_outputs, test_targets = torch.cat(test_outputs), torch.cat(test_targets)

test_accuracy = torchmetrics.functional.accuracy(test_outputs, test_targets, task="binary")
test_mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(test_outputs, test_targets)

print(f"\tTest accuracy: {test_accuracy*100.0:.1f}%")
print(f"\tTest MCC: {test_mcc*100.0:.1f}%")


fpr, tpr, thresholds = roc_curve(test_targets, test_outputs)
auc = roc_auc_score(test_targets, test_outputs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig(outputs_folder + "/roc.png")


# Selected threshold as 0.5
test_outputs = (test_outputs>0.5).float()
metric = BinaryConfusionMatrix()
print('\t' + str(metric(test_outputs, test_targets).cpu().detach().numpy()).replace('\n', '\n\t'))