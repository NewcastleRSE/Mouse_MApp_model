import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="pre-trained model", required=True)
parser.add_argument(
    "--resume", action="store_true", help="resume from most recent checkpoint"
)
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--dropout_level", type=float, default=0.4, help="dropout")
parser.add_argument("--w_decay", type=float, default=0.00001, help="weight decay")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--seed_n", type=int, default=74, help="random seed")
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

random.seed(opt.seed_n)
np.random.seed(opt.seed_n)
torch.manual_seed(opt.seed_n)
torch.cuda.manual_seed(opt.seed_n)

num_classes = 4

result_path = f"{Path.home()}/Data/Mouse/Results/"
pretrain_file = opt.model + "_pretrain_model_state.cpt"


def set_parameter_requires_grad(model):
    """Freeze model Gradiends
    Arguments:
        model (object): pytorch model
    """
    for param in model.parameters():
        param.requires_grad = True


def get_model(
    model_name=opt.model, num_classes=num_classes, lr=opt.lr, w_decay=opt.w_decay
):
    """Get pre-trained model with additional layer suitable for num_classes
    Arguments:
        model_name (string): model to load
        num_classes: (int): number of unique labels/classes
        lr (float): learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "alexnet":
        model = models.alexnet(pretrained=False)
        model.features[0] = nn.Conv2d(
            1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)
        )
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg19":
        model = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(11, 11), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)
        )
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = num_classes

    elif model_name == "densenet201":
        model = models.densenet201(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        model = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    return model


def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert actual.size(0) == predicted.size(0)
    return float(actual.eq(predicted).sum()) / actual.size(0)


def train_class(trainloader, model):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(opt.n_epochs):
        training_loss = 0
        model.train()
        accuracy = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            _, predicted = torch.max(logps.data, 1)

            accuracy += (predicted == labels).sum().item()

        print(
            "\tEpoch {:>3}/{:<3} | Train loss: {:>6.4f}  ".format(
                epoch + 1,
                opt.n_epochs,
                training_loss / len(trainloader),
            )
        )

    return epoch, model, optimizer


def test_class(testloader, model):
    test_loss = 0
    accuracy = 0
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for valid_inputs, valid_labels in testloader:
            valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(
                device
            )

            logps = model(valid_inputs)

            batch_loss = criterion(logps, valid_labels)
            test_loss += batch_loss.item()

            _, predicted = torch.max(logps.data, 1)
            accuracy += (predicted == valid_labels).sum().item()

    test_acc = accuracy / len(testloader)
    print("Test Accuracy: %2.5f" % (accuracy / len(testloader)))

    return test_acc

def save_model(epoch, model, optimizer, filepath=pretrain_file):
    """Save the model and embeddings"""

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(model, filepath)
    print("State Saved")


class MouseDataset(Dataset):
    def __init__(self, df, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["img_path"].iloc[idx]
        img_name = self.df["filename"].iloc[idx]

        img_path = f"{Path.home()}/Data/Mouse/preprocessing/{img_path}/{img_name}"
        img = Image.open(img_path)
        img = img.convert("L")
        img = img.convert("RGB")
        img_label = self.df.mouse_score.iloc[idx]

        if img_label == 2.5 or img_label == 2.0:
            img_label = 0
        elif img_label == 3.0 or img_label == 3.5:
            img_label = 1
        elif img_label == 4.0 or img_label == 4.5:
            img_label = 2
        elif img_label == 5.0:
            img_label = 3
        else:
            print("LABELS ARE BROKEN")

        if self.transform:
            img = self.transform(img)

        return img, img_label


if __name__ == "__main__":
    crop_size = 224

    flip = transforms.RandomHorizontalFlip()
    crop = transforms.RandomCrop(crop_size)
    shear = transforms.RandomAffine(
        degrees=0, translate=None, scale=None, shear=20
    )  # 20 degrees random shear

    train_transforms = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            flip,
            crop,
            shear,
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    )

    csv_path = "sample_data/df.csv"
    csv_pd = pd.read_csv(csv_path)
    
    df = csv_pd.sort_values("mouse_score")
    groups = df.groupby(["mouse_score", "mouse_id"])

    group_keys = list(groups.groups.keys())
    np.random.shuffle(group_keys)

    train_ids, test_ids = train_test_split(group_keys, test_size=0.2)
    train_df = pd.concat([groups.get_group(id) for id in train_ids])
    test_df = pd.concat([groups.get_group(id) for id in test_ids])

    train_dataset = MouseDataset(train_df, train_transforms)
    test_dataset = MouseDataset(test_df, test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    model = get_model(
        model_name=opt.model, num_classes=num_classes, lr=opt.lr, w_decay=opt.w_decay
    )

    if opt.resume:
        print("resume")
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        state = torch.load(pretrain_file)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
    else:
        print("training from beginning")

    epoch, model, optimizer = train_class(trainloader, model)
    # epoch, model, optimizer = train_class(trainloader, testloader, model)

    save_model(epoch, model, optimizer)

    model = torch.load(pretrain_file)

    test_acc = test_class(testloader, model)
    