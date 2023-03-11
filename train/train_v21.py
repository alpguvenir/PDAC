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

from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MulticlassMatthewsCorrCoef


from dataset import Dataset_radimagenet2
from models.ResNet18.ResNet18_rad_v2 import ResNet18_rad_v2


def get_file_paths(path):
    return glob.glob(path + "/*")


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


# Getting the current date and time
dt = datetime.now()
ts = int(datetime.timestamp(dt))


torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


NUM_EPOCHS = 100
BATCH_SIZE = 32
lr = 0.001


#version = "radimagenet_v1" # onehot encoded
version = "radimagenet_v2" # class index 


outputs_folder = "../../results-train/" + "radimagenet" + "-" + str(ts)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)


file = open("../dataset/" + version + "/radimagenet_0.8_train_ct_scans_list.pickle",'rb')
train_ct_scans_list = pickle.load(file)
file.close()

file = open("../dataset/" + version + "/radimagenet_0.8_train_ct_labels_list.pickle",'rb')
train_ct_labels_list = pickle.load(file)
file.close()

file = open("../dataset/" + version + "/radimagenet_0.1_val_ct_scans_list.pickle",'rb')
val_ct_scans_list = pickle.load(file)
file.close()

file = open("../dataset/" + version + "/radimagenet_0.1_val_ct_labels_list.pickle",'rb')
val_ct_labels_list = pickle.load(file)
file.close()

file = open("../dataset/" + version + "/radimagenet_0.1_test_ct_scans_list.pickle",'rb')
test_ct_scans_list = pickle.load(file)
file.close()

file = open("../dataset/" + version + "/radimagenet_0.1_test_ct_labels_list.pickle",'rb')
test_ct_labels_list = pickle.load(file)
file.close()



with open(outputs_folder + "/log.txt", "w") as file:
    print("Class label", params_dict.get("data.label.name"))
    file.write("Class label " + params_dict.get("data.label.name") + "\n")

    print("Class shuffle version", params_dict.get("data.label.name.version"))
    file.write("Class shuffle version " + params_dict.get("data.label.name.version") + "\n")

    file.close()



# Transforms that need to be applied to the dataset
transforms = {
                'Resize': {'height': 256, 'width': 256},    # Original CT layer sizes are 512 x 512
            }


train_dataset = Dataset_radimagenet2.Dataset_radimagenet2(train_ct_scans_list, train_ct_labels_list, transforms=transforms)
val_dataset = Dataset_radimagenet2.Dataset_radimagenet2(val_ct_scans_list, val_ct_labels_list, transforms=transforms)
test_dataset = Dataset_radimagenet2.Dataset_radimagenet2(test_ct_scans_list, test_ct_labels_list, transforms=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18_rad_v2()
model.to(device)


get_params = lambda m: sum(p.numel() for p in m.parameters())
with open(outputs_folder + "/log.txt", "a") as file:
    print(f"Complete model has {get_params(model)} params")
    file.write("Complete model has " + str(get_params(model)) + " params" + "\n")
    file.close()


# loss and optimizer
confmat = ConfusionMatrix(task="multiclass", num_classes=28)
metric = MulticlassMatthewsCorrCoef(num_classes=28)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr)
sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.01)


train_loss_array = []
train_accuracy_array = []
train_mcc_array = []

validation_loss_array = []
validation_accuracy_array = []
validation_mcc_array = []

test_loss_array = []
test_accuracy_array = []
test_mcc_array = []



for epoch in range(NUM_EPOCHS):
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"Epoch: {epoch}")
        file.write("Epoch: " + str(epoch) + "\n")
        file.close()
    
    total_train_loss = 0
    model.train()

    
    #### TRAIN ####
    pbar_train_loop = tqdm(train_loader, total=len(train_loader), leave=False)

    train_outputs= []
    train_targets= []

    train_iteration = 0
    # torch.Size([1, 256, 256, 1]) torch.Size([1, 28])
    for train_input, train_target in pbar_train_loop:
        optimizer.zero_grad()
        
        train_input = train_input.permute(0, 3, 1, 2)

        train_output = model(train_input.to(device))

        train_loss = criterion(train_output, train_target.to(device))
        train_loss.backward()

        optimizer.step()

        train_loss_value = train_loss.detach().cpu().item()
        total_train_loss += train_loss_value
        pbar_train_loop.set_description_str(f"Loss: {train_loss_value:.2f}")

        train_outputs.append(train_output.cpu())
        train_targets.append(train_target)

        train_iteration += 1

    sched.step()

    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\tMean train loss: {total_train_loss / len(train_loader):.2f}")
        file.write("\tMean train loss: " +  "{:.2f}".format(total_train_loss / len(train_loader)) + "\n")
    
        train_outputs, train_targets = torch.cat(train_outputs), torch.cat(train_targets)
        
        train_accuracy = torchmetrics.functional.accuracy(train_outputs, train_targets, num_classes=28, task="multiclass")
        train_mcc = metric(train_outputs, train_targets)
        

        print(f"\tTrain accuracy: {train_accuracy*100.0:.1f}%    Train MCC: {train_mcc*100.0:.1f}%")
        file.write("\tTrain accuracy: " + "{:.1f}".format(train_accuracy*100.0) + "%    Train MCC: " + "{:.1f}".format(train_mcc*100.0) + "%" + "\n")


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

            validation_input = validation_input.permute(0, 3, 1, 2)

            validation_output = model(validation_input.to(device))

            validation_loss = criterion(validation_output, validation_target.to(device))

            validation_loss_value = validation_loss.detach().cpu().item()
            total_validation_loss += validation_loss_value

            validation_outputs.append(validation_output.cpu())
            validation_targets.append(validation_target)
    
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\n\tMean validation loss: {total_validation_loss / len(val_loader):.2f}")
        file.write("\n\tMean validation loss: " +  "{:.2f}".format(total_validation_loss / len(val_loader)) + "\n")
                
        validation_outputs, validation_targets = torch.cat(validation_outputs), torch.cat(validation_targets)
        
        validation_accuracy = torchmetrics.functional.accuracy(validation_outputs, validation_targets, num_classes=28, task="multiclass")
        validation_mcc = metric(validation_outputs, validation_targets)

        print(f"\tValidation accuracy: {validation_accuracy*100.0:.1f}%    Validation MCC: {validation_mcc*100.0:.1f}%")    
        file.write("\tValidation accuracy: " + "{:.1f}".format(validation_accuracy*100.0) + "%    Validation MCC: " + "{:.1f}".format(validation_mcc*100.0) + "%" + "\n")

    
    validation_loss_array.append(total_validation_loss / len(val_loader))
    validation_accuracy_array.append(validation_accuracy)
    validation_mcc_array.append(validation_mcc)


    #### TEST ####
    model.eval()

    total_test_loss = 0
    test_outputs = []
    test_targets = []

    test_iteration = 0
    with torch.no_grad():
        for test_input, test_target in tqdm(test_loader, leave=False):

            test_input = test_input.permute(0, 3, 1, 2)

            test_output = model(test_input.to(device))

            test_loss = criterion(test_output, test_target.to(device))

            test_loss_value = test_loss.detach().cpu().item()
            total_test_loss += test_loss_value

            test_outputs.append(test_output.cpu())
            test_targets.append(test_target)
        
            test_iteration += 1
    
    with open(outputs_folder + "/log.txt", "a") as file:
        print(f"\n\tMean test loss: {total_test_loss / len(test_loader):.2f}")
        file.write("\n\tMean test loss: " +  "{:.2f}".format(total_test_loss / len(test_loader)) + "\n")
    
        test_outputs, test_targets = torch.cat(test_outputs), torch.cat(test_targets)
        
        test_accuracy = torchmetrics.functional.accuracy(test_outputs, test_targets, num_classes=28, task="multiclass")
        test_mcc = metric(test_outputs, test_targets)

        print(f"\tTest accuracy: {test_accuracy*100.0:.1f}%    Test MCC: {test_mcc*100.0:.1f}%")
        file.write("\tTest accuracy: " + "{:.1f}".format(test_accuracy*100.0) + "%    Test MCC: " + "{:.1f}".format(test_mcc*100.0) + "%" + "\n")


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

    #### SAVE MODEL WEIGHTS ####
    path_model = outputs_folder + "/model" + str(epoch) + ".pth"
    torch.save(model.state_dict(), path_model)

    with open(outputs_folder + "/log.txt", "a") as file:
        print("\n")
        file.write("\n")
        file.close()