from typing import Any

from PIL import Image
import numpy as np
import glob
import json

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn


def getlabel(data_json):
    data_label = []
    for i in data_json:
        data_label.append(data_json[i]['label'])
    return data_label


class SVHNDataset(Dataset):  # Street View House Number
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]
        # print torch.from_numpy(np.array(lbl[:5]))
        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)


class SVHNModel1(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(SVHNModel1, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 )
        self.fc1 = nn.Linear(32 * 3 * 7, 11)
        self.fc2 = nn.Linear(32 * 3 * 7, 11)
        self.fc3 = nn.Linear(32 * 3 * 7, 11)
        self.fc4 = nn.Linear(32 * 3 * 7, 11)
        self.fc5 = nn.Linear(32 * 3 * 7, 11)
        self.fc6 = nn.Linear(32 * 3 * 7, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6


def train(data_loader, model, criterion, optimizer):
    train_loss = []
    model.train()
    for i, (input, target) in enumerate(data_loader):
        target = target.long()
        cc0, cc1, cc2, cc3, cc4, cc5 = model(input)
        loss = criterion(cc0, target[:, 0]) + \
            criterion(cc1, target[:, 1]) + \
            criterion(cc2, target[:, 2]) + \
            criterion(cc3, target[:, 3]) + \
            criterion(cc4, target[:, 4]) + \
            criterion(cc5, target[:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print(i)
        #     print(loss.item())
        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(data_loader, model, criterion, optimizer):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            target = target.long()
            cc0, cc1, cc2, cc3, cc4, cc5 = model(input)
            loss = criterion(cc0, target[:, 0]) + \
                criterion(cc1, target[:, 1]) + \
                criterion(cc2, target[:, 2]) + \
                criterion(cc3, target[:, 3]) + \
                criterion(cc4, target[:, 4]) + \
                criterion(cc5, target[:, 5])
            loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(data_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    for j in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(data_loader):
                target = target.long()
                cc0, cc1, cc2, cc3, cc4, cc5 = model(input)
                output = np.concatenate([
                    cc0.data.numpy(),
                    cc1.data.numpy(),
                    cc2.data.numpy(),
                    cc3.data.numpy(),
                    cc4.data.numpy(),
                    cc5.data.numpy()], axis=1)
                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    return test_pred_tta


train_path = glob.glob('input/train/*.png')
train_path.sort()
train_json = json.load(open('input/train.json'))
train_label = getlabel(train_json)
# num = 0
# for i in train_label:
#     if len(i) == 0:
#         num += 1
# print(num)

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])), batch_size=40, shuffle=False)

# for data in train_loader:
#      print(data)

val_path = glob.glob("input/validation/*.png")
val_json = json.load(open("input/validation.json"))
val_label = getlabel(val_json)

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])), batch_size=40, shuffle=False)

model1 = SVHNModel1()
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), 0.001)
best_loss = 1000.0
for epoch in range(10):
    train_loss = train(train_loader, model1, criterion1, optimizer1)
    val_loss = validate(val_loader, model1, criterion1, optimizer1)
    # print(val_loader.dataset)
    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model1, 1)
    val_predict_label = np.array(val_predict_label)
    # print(val_predict_label)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
        val_predict_label[:, 55:66].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))
    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
    print('Epoch: {0}, Train loss: {1} \t Val loss: {2} \t acc: {3}'.format(epoch, train_loss, val_loss, val_char_acc))
    if val_loss < best_loss:
        best_loss = val_loss
    torch.save(model1.state_dict(), './model.pt')
