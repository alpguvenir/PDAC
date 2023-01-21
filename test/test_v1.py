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
from models.ResNet18.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNet18.ResNet18_sum import ResNet18_sum
from models.ResNet18.ResNet18_mean import ResNet18_mean
from models.ResNet18.ResNet18_Linear import ResNet18_Linear

from models.ResNet101.ResNet101_MultiheadAttention import ResNet101_MultiheadAttention
from models.ResNet101.ResNet101_sum import ResNet101_sum


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
                'Max-Layers' : {'max': 100},
                
                'uniform-number-of-layers' : {'bool': False},
                'Uniform-Layers': {'uniform': 200},
            }




test_dataset = Dataset.Dataset(test_ct_scans_list, test_ct_labels_list, transforms=transforms, train_mode = False)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# Use gpu if exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = ResNet101_MultiheadAttention()
model.load_state_dict(torch.load('/home/guevenira/attention_CT/PDAC/results-train/befund_verlauf-1674124609/model4.pth'))
model.to(device)


# loss and optimizer
sigmoid = nn.Sigmoid()
criterion = nn.BCEWithLogitsLoss()


#### TEST ####
model.eval()

total_test_loss = 0
test_outputs = []
test_targets = []

sampled1 = False
sampled0 = False

with torch.no_grad():
    for test_input, test_target in tqdm(test_loader, leave=False):

        test_input = test_input.permute(2, 0, 1, 3, 4)
        test_input = torch.squeeze(test_input, 1)

        # Use only with models with attention
        #test_output, att_map = model(test_input.to(device))
        test_output = model(test_input.to(device))

        test_loss = criterion(test_output, torch.unsqueeze(test_target, 0).float().to(device))

        test_loss_value = test_loss.detach().cpu().item()

        total_test_loss += test_loss_value

        test_outputs.append(sigmoid(test_output.cpu().flatten()))
        test_targets.append(test_target.flatten())

        
        if test_input.shape[0] == 200:

            if not(sampled1) and (test_target.flatten()[0] == 1 and sigmoid(test_output.cpu().flatten()) > 0.5):
                
                # Use only with models with attention
                # Begin
                """
                top5 = torch.topk(att_map.flatten(), 5).indices
                for top_index in range(len(top5)):
                    cl_layer = test_input[top5[top_index], :, :, :]                
                    cl_layer = cl_layer.permute(1, 2, 0)
                    plt.imshow(cl_layer, cmap='gray')
                    plt.savefig(outputs_folder + "/1-pred-1_" + str(top_index) + ".png")
                    plt.close()

                
                y = att_map.cpu().detach().numpy()
                y = y[0][0]
                x = list(range(0, len(y)))
                plt.plot(x, y)
                plt.savefig(outputs_folder + "/plot1.png")
                plt.close()
                print("class 1", top5)
                """
                # End

                cl_layer = test_input[0, :, :, :]                
                cl_layer = cl_layer.permute(1, 2, 0)
                plt.imshow(cl_layer, cmap='gray')
                plt.savefig(outputs_folder + "/1-pred-1_layer_0th.png")
                plt.close()

                cl_layer = test_input[199, :, :, :]                
                cl_layer = cl_layer.permute(1, 2, 0)
                plt.imshow(cl_layer, cmap='gray')
                plt.savefig(outputs_folder + "/1-pred-1_layer_199th.png")
                plt.close()
                sampled1 = True

            
            if not(sampled0) and (test_target.flatten()[0] == 0 and sigmoid(test_output.cpu().flatten()) < 0.5):
                
                # Use only with models with attention
                # Begin
                """
                top5 = torch.topk(att_map.flatten(), 5).indices
                for top_index in range(len(top5)):
                    cl_layer = test_input[top5[top_index], :, :, :]                
                    cl_layer = cl_layer.permute(1, 2, 0)
                    plt.imshow(cl_layer, cmap='gray')
                    plt.savefig(outputs_folder + "/0-pred-0_" + str(top_index) + ".png")
                    plt.close()

                y = att_map.cpu().detach().numpy()
                y = y[0][0]
                x = list(range(0, len(y)))
                plt.plot(x, y)
                plt.savefig(outputs_folder + "/plot0.png")
                plt.close()
                print("class 0", top5)
                """
                # End

                cl_layer = test_input[0, :, :, :]                
                cl_layer = cl_layer.permute(1, 2, 0)
                plt.imshow(cl_layer, cmap='gray')
                plt.savefig(outputs_folder + "/0-pred-0_layer_0th.png")
                plt.close()

                cl_layer = test_input[199, :, :, :]                
                cl_layer = cl_layer.permute(1, 2, 0)
                plt.imshow(cl_layer, cmap='gray')
                plt.savefig(outputs_folder + "/0-pred-0_layer_199th.png")
                plt.close()
                sampled0 = True
            

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