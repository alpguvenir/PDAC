import sys 
# append the path to folder /train/ in order to use other folders dataset, model...
sys.path.append('/home/guevenira/attention_CT/PDAC/src/')

import yaml
import glob
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix

from dataset import Dataset
from models.ResNet18.ResNet18_MultiheadAttention import ResNet18_MultiheadAttention
from models.ResNet34.ResNet34_MultiheadAttention import ResNet34_MultiheadAttention
from models.ResNet50.ResNet50_MultiheadAttention import ResNet50_MultiheadAttention


def get_file_paths(path):
    return glob.glob(path + "/*")


torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


with open('../parameters.yml') as params:
    params_dict = yaml.safe_load(params)


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


NUM_EPOCHS = 20
BATCH_SIZE = 1
lr = 0.001


train_dataset = Dataset.Dataset(train_ct_scans_list, train_ct_labels_list, transforms=transforms)
val_dataset = Dataset.Dataset(val_ct_scans_list, val_ct_labels_list, transforms=transforms)
test_dataset = Dataset.Dataset(test_ct_scans_list, test_ct_labels_list, transforms=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# initialize model
model = ResNet34_MultiheadAttention()
sigmoid = nn.Sigmoid()


get_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"Complete model has {get_params(model)} params")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)


# loss and optimize
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.01)
weight_raito = train_ct_labels_list.count(0) / train_ct_labels_list.count(1)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    total_loss = 0
    model.train()

    if epoch == 0:
        model.feature_extractor.eval()
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        model.feature_extractor.conv1[0].weight.requires_grad = True
        model.feature_extractor.conv1[0].bias.requires_grad = True
    elif epoch == 1:
        model.feature_extractor.eval()
    elif epoch == 2:
        pass
    elif epoch == 3:
        for param in model.feature_extractor.layer4.parameters():
            param.requires_grad = True
    elif epoch == 4:
        for param in model.feature_extractor.layer3.parameters():
            param.requires_grad = True
    elif epoch > 4:
        for param in model.parameters():
            param.requires_grad = True
    
    pbar_train_loop = tqdm(train_loader, total=len(train_loader), leave=False)
    
    for input, target in pbar_train_loop:
        optimizer.zero_grad()

        input = input.permute(2, 0, 1, 3, 4)
        input = torch.squeeze(input, 1)

        out = model(input.to(device))

        # https://medium.com/dataseries/how-to-deal-with-unbalanced-dataset-in-binary-classification-part-3-e580d8d09883
        loss = criterion(out, torch.unsqueeze(target, 0).float().to(device))

        weighted_loss = (target.detach().cpu().item() * (weight_raito - 1)) + 1
        # print(loss * weighted_loss)

        loss = loss * weighted_loss

        loss.backward()
        optimizer.step()
        lv = loss.detach().cpu().item()
        total_loss += lv
        pbar_train_loop.set_description_str(f"Loss: {lv:.2f}")
    
    print(f"\tMean train loss: {total_loss / len(train_loader):.2f}")
    sched.step()
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for input, target in tqdm(val_loader, leave=False):

            input = input.permute(2, 0, 1, 3, 4)
            input = torch.squeeze(input, 1)

            out = model(input.to(device))

            preds.append(sigmoid(out.cpu().flatten()))
            targets.append(target.flatten())
    preds, targets = torch.cat(preds), torch.cat(targets)
    acc = torchmetrics.functional.accuracy(preds, targets, task="binary")
    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(preds, targets,)

    print(f"\tVal accuracy: {acc*100.0:.1f}%")
    print(f"\tVal MCC: {mcc*100.0:.1f}%")

    preds = (preds>0.5).float()
    metric = BinaryConfusionMatrix()
    print('\t' + str(metric(preds, targets).cpu().detach().numpy()).replace('\n', '\n\t'))


    preds, targets = [], []
    with torch.no_grad():
        for input, target in tqdm(test_loader, leave=False):

            input = input.permute(2, 0, 1, 3, 4)
            input = torch.squeeze(input, 1)

            out = model(input.to(device))

            preds.append(sigmoid(out.cpu().flatten()))
            targets.append(target.flatten())
    preds, targets = torch.cat(preds), torch.cat(targets)
    acc = torchmetrics.functional.accuracy(preds, targets, task="binary")
    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(preds, targets,)

    print(f"\tTest accuracy: {acc*100.0:.1f}%")
    print(f"\tTest MCC: {mcc*100.0:.1f}%")

    preds = (preds>0.5).float()
    metric = BinaryConfusionMatrix()
    print('\t' + str(metric(preds, targets).cpu().detach().numpy()).replace('\n', '\n\t'))

    print("\n")