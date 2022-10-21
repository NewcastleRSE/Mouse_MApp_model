import os
import argparse

import numpy as np
import random
import torch
import sys

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="pre-trained model", required=True)
parser.add_argument(
    "--resume", action="store_true", help="resume from most recent checkpoint"
)
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs")
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

# setup tensorboard stuff
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}_{opt.model}")

result_path = f"{Path.home()}/Data/Mouse/Results/"
pretrain_file = opt.model + "_pretrain_model_state.cpt"


def check_valid(path):
    path = Path(path)
    return not path.stem.startswith(".")


def plot_error_matrix(cm, classes, cmap=plt.cm.Blues):
    """Plot the error matrix for the neural network models"""

    from sklearn.metrics import confusion_matrix
    import itertools

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()


def CCM(cnf_labels, cnf_predictions):

    class_names = ["2", "3", "4", "5"]
    cnf_matrix = confusion_matrix(cnf_labels, cnf_predictions)
    np.set_printoptions(precision=2)

    # Normalise
    cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({"font.size": 16})

    # Plot normalized confusion matrix
    plt.figure()
    plot_error_matrix(cnf_matrix, classes=class_names)
    plt.tight_layout()
    filename = "alexnet_nweight.pdf"
    plt.savefig(filename, format="PDF", bbox_inches="tight")
    plt.show()


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
        model = models.vgg19(pretrained=True)
        model.features[0] = nn.Conv2d(
            1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)
        )
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

    ntest = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    label_list = []
    prediction_list = []

    with torch.no_grad():
        for valid_inputs, valid_labels in testloader:
            valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(
                device
            )

            images = valid_inputs.cpu()
            images = images[0].numpy().transpose(1, 2, 0)
            norm_image = cv2.normalize(
                images,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

            norm_image = norm_image.astype(np.uint8)
            scale_percent = 220  # percent of original size
            width = int(norm_image.shape[1] * scale_percent / 100)
            height = int(norm_image.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized_img = cv2.resize(norm_image, dim, interpolation=cv2.INTER_CUBIC)

            gt = valid_labels.cpu() + 2

            logps = model(valid_inputs)

            batch_loss = criterion(logps, valid_labels)
            test_loss += batch_loss.item()

            _, predicted = torch.max(logps.data, 1)
            accuracy += (predicted == valid_labels).sum().item()

            pred = predicted.cpu() + 2

            target_label = valid_labels.cpu()
            pred_label = predicted.cpu()

            label_list.append(target_label)
            prediction_list.append(pred_label)

            ntest = ntest + 1

    cnf_labels = np.array(label_list)
    cnf_predictions = np.array(prediction_list)

    test_acc = accuracy / len(testloader)
    print("Test Accuracy: %2.5f" % (accuracy / len(testloader)))

    return test_acc, cnf_labels, cnf_predictions


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

        img_path = f"{Path.home()}{img_path}/{img_name}"
        img = Image.open(img_path)
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

    main_path = f"{Path.home()}/Data/Sample/"

    # getting the csv files
    ext = "csv"
    count = 1
    df = []
    for subdir, dirs, files in os.walk(main_path + "train"):
        for i in files:
            if i.endswith(ext):
                csv_path = subdir + "/" + i
                csv_pd = pd.read_csv(csv_path)
                if count == 1:
                    df = csv_pd
                elif count > 1:
                    df = pd.concat([df, csv_pd])
                count += 1
    train_df = df

    ext = "csv"
    count = 1
    df = []
    for subdir, dirs, files in os.walk(main_path + "test"):
        for i in files:
            if i.endswith(ext):
                csv_path = subdir + "/" + i
                csv_pd = pd.read_csv(csv_path)
                if count == 1:
                    df = csv_pd
                elif count > 1:
                    df = pd.concat([df, csv_pd])
                count += 1
    test_df = df

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
    # state = torch.load(pretrain_file)
    # model.load_state_dict(state["state_dict"])
    # model.load_state_dict(state['state_dict'])
    # model.eval()

    test_acc, cnf_labels, cnf_predictions = test_class(testloader, model)
    CCM(cnf_labels, cnf_predictions)
