import glob
import json

import datetime
import numpy as np
# %pylab inline
import torch
from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ##用于指定用哪块GPU
# import cv2

torch.manual_seed(0)  # 限定一个随机种子点，保证每次运行结果相同，方便验证

# # 使用GPU需用用到cuda模块，配合cudnn加速模块一起使用
torch.backends.cudnn.deterministic = True
# # 用以保证实验的可重复性
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

name_file = "demo2"
print(name_file)

train_path = glob.glob('input/train_augmentation/*.png')
# glob函数用于模糊匹配一定格式的文件名
train_path.sort()
train_json = json.load(open('input/augmentation.json'))
train_label = [train_json[x]['label'] for x in train_json]


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')  # 转成RGB格式图片

        if self.transform is not None:
            img = self.transform(img)  # 把图片裁剪缩放到指定形式

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]  # 标签扩展到统一长度
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)


class SVHN_Model(nn.Module):
    def __init__(self):
        super(SVHN_Model, self).__init__()
        model_conv = models.resnet18(pretrained=True)
        # #加载模型，并设为预训练模式
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        # 自适应平均池化，strides，paddings等参数都自适应好了
        # model_conv.dropout = nn.Dropout(0.4)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        # 取模型结果
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        # nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 11)
        # nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 11)
        # nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 11)
        # nn.Dropout(0.5)
        self.fc5 = nn.Linear(512, 11)
        # nn.Dropout(0.5)
        # 五个fc层并联，由于图片对应的是一个5位数的值，分别对每一个进行识别，所以5个fc

    def forward(self, img):
        feat = \
            self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5


train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),  # 统一缩放到64*128
                    transforms.RandomCrop((60, 120)),  # 随机剪裁
                    transforms.ColorJitter(0.3, 0.3, 0.2),  # 调整亮度透明度
                    transforms.RandomRotation(10),  # 随机旋转
                    transforms.ToTensor(),  # 转成一个tensor
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
                ])),
    batch_size=64,  # 一个batch的图片数量
    shuffle=True,
    num_workers=0,  # num_workers为线程数目，之前为10，报错，由于线程数与内核数不匹配，把num_workers改为0成功
)

val_path = glob.glob('input/validation/*.png')
val_path.sort()
val_json = json.load(open('input/validation.json'))
val_label = [val_json[x]['label'] for x in val_json]
# print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([  # 图像预处理常写在compose内，成为一组操作
                    transforms.Resize((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用resnet18必须标准化到特定
                ])),
    batch_size=40,
    shuffle=False,
    num_workers=0,
)


def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):  # 每次循环为一个batch
        target = target.long()
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])
        # 计算损失函数的时候是5个FC层的损失相加

        optimizer.zero_grad()
        loss.backward()  # 损失回传
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.long()
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                target = target.long()
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),  # 预测阶段只用cpu就可以
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


model = SVHN_Model()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), 0.0008)  # 0.001为学习率
best_loss = 1000.0
time = datetime.datetime.now()

# 是否使用GPU
use_cuda = True
if use_cuda:
    model = model.cuda()

print('GPU: {0}'.format(use_cuda))
for epoch in range(20):
    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)
    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

    print('Epoch: {0}  Train loss: {1}  Val loss: {2}  val_acc: {3}'.format(epoch, train_loss, val_loss, val_char_acc))
    with open("log/{0}.txt".format(time.date()), "a") as fp:
        fp.write(
            'Mod: {0} V: {1} Epoch: {2} Train loss: {3}Val loss: {4} Val_acc: {5} time: {6} \n'.format("model2",
                                                                                                       name_file,
                                                                                                       epoch,
                                                                                                       train_loss,
                                                                                                       val_loss,
                                                                                                       val_char_acc,
                                                                                                       time))
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model, 'mod/model2_val acc{0}_{1}.pt'.format(val_char_acc, name_file))
