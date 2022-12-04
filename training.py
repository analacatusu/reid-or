from __future__ import print_function, division

import numpy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from models import *
from transformations import Rescale, RandomVerticalFlip, RandomHorizontalFlip, Normalize, ToTensor
import torch.optim as optim
from pathlib import Path
from dataset import ReIDDataset, show_batch, debug
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from pytorch_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def matplotlib_imshow(img):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(device)
    x = img * STD[:, None, None] + MEAN[:, None, None]
    x = x.cpu().numpy().transpose(1,2,0)
    pil_image = Image.fromarray(x.astype('uint8'), 'RGB')
    pil_image = pil_image.resize((50,50))
    plt.imshow(pil_image)


def plot(val_imgs, val_preds, val_ids):
    fig = plt.figure(figsize=(10, 12))
    for idx in numpy.arange(5):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(val_imgs[idx])
        ax.set_title("pred:{0}, label:{1}".format(
            val_preds[idx],
            val_ids[idx]),
            color=("green" if val_preds[idx] == val_ids[idx].item() else "red"))
    return fig


if __name__ == '__main__':
    torch.cuda.empty_cache()
    ids_and_names = {
        0: "Evin",
        1: "Tianyu",
        2: "Lennart",
        3: "Ege",
        4: "Chantal"
    }
    list_of_classes = [0,1,2,3,4]
    log_dir_path = Path("/tmp/pycharm_project_751/log")
    writer = SummaryWriter(log_dir=str(log_dir_path))
    project_path = Path("/tmp/pycharm_project_751")
    root_path = Path("/tmp/pycharm_project_751/inputs")
    train_path = root_path / "train"
    val_path = root_path / "val"
    datasets = {}
    reid_dataset_train = ReIDDataset(root_dir=str(train_path),
                                     transform=transforms.Compose([
                                         Rescale(128),
                                         RandomVerticalFlip(0.5),
                                         RandomHorizontalFlip(0.9),
                                         ToTensor(),
                                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ]))

    print(len(reid_dataset_train))

    hparams = {
        "batch_size_train": 32,
        "batch_size_val": 32,
        "lr": 0.0002,
        "weight_decay": 5e-4,
        "nr_epochs": 8,
        "num_classes": 5
    }

    dataloader_train = DataLoader(reid_dataset_train, batch_size=hparams['batch_size_train'], shuffle=True, num_workers=2, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model, classifier = build_model(num_classes=hparams['num_classes'])

    model.to(device)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(parameters, lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    train_loss_history = []  # loss
    train_acc_history = []  # accuracy

    for epoch in range(hparams['nr_epochs']):

        running_loss = 0.0
        correct = 0.0
        total = 0
        val_ids = []
        val_preds = []
        val_imgs = []

        model.train()
        torch.backends.cudnn.enabled = False

        with tqdm(dataloader_train, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):

                tepoch.set_description(f"Training Epoch {epoch + 1}")
                imgs = data['image'].to(device)
                ids = torch.squeeze(data['id'].to(device))

                optimizer.zero_grad()

                features = model(imgs)
                outputs = classifier(features)

                loss = criterion(outputs, ids)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)  # convert output probabilities of each class to a singular class prediction

                if i == 0:

                    val_preds = preds[0:5]
                    val_imgs = imgs[0:5]
                    val_ids = ids[0:5]

                correct += preds.eq(ids).sum().item()
                total += ids.size(0)

                correct_ones = preds.eq(ids)

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * (correct / total))

        running_loss /= len(dataloader_train)
        correct /= total
        print("[Epoch %d] train_loss: %.3f train_acc: %.2f %%" % (epoch + 1, running_loss, 100 * correct))

        train_loss_history.append(running_loss)
        train_acc_history.append(correct)
        writer.add_scalar("Loss/train", running_loss, epoch + 1)
        writer.add_scalar("Acc/train", 100 * correct, epoch + 1)
        writer.add_figure('prediction vs. actual', plot(val_imgs, val_preds, val_ids), epoch + 1)

    with open("reid_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("reid_classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)

    writer.flush()
    writer.close()

    plt.plot(train_acc_history)
    plt.plot(train_loss_history)
    plt.title(f"SimpleReID - {hparams['nr_epochs']} epochs training")
    plt.xlabel('iteration')
    plt.ylabel('acc/loss')
    plt.legend(['acc-train', 'loss-train', 'acc-val', 'loss-val'])
    plt.show()



