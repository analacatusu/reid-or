import os.path
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from transformations import SquarePad2
from dataset2 import ReIDDataset
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy
import pickle
import torchmetrics
from PIL import Image
import cv2
from grad_cam import grad_cam


ids_and_names = {
        0: "Evin",
        1: "Tianyu",
        2: "Lennart",
        3: "Ege",
        4: "Chantal"
    }

take_1 = [0 for k in range(5)]
take_1_done = [1 for m in range(5)]
take_5 = [0 for l in range(5)]
take_5_done = [1 for n in range(5)]
image_paths_wrong = []
names_wrong = []
predicts_wrong = []
image_paths_correct = []
names_correct = []


def correctly_classified(im_paths, correct_ones, ids, preds=None):
    for i, id in enumerate(ids):
        if correct_ones[i]:
            take_nr = im_paths[i].split('/')[-4]
            print(f"{take_nr=}")
            take_nr = take_nr.split('_')[-1]
            print(f"{take_nr=}")
            take_nr = int(take_nr)
            if take_nr == 1 and take_1[id.item()] == 0:
                take_1[id.item()] = 1
                image_paths_correct.append(im_paths[i])
                names_correct.append(ids_and_names[id.item()])
            if take_nr == 5 and take_5[id.item()] == 0:
                take_5[id.item()] = 1
                image_paths_correct.append(im_paths[i])
                names_correct.append(ids_and_names[id.item()])


def wrongly_classified(im_paths, correct_ones, ids, preds=None):
    for i, id in enumerate(ids):
        if not correct_ones[i]:
            take_nr = im_paths[i].split('/')[-4]
            print(f"{take_nr=}")
            take_nr = take_nr.split('_')[-1]
            print(f"{take_nr=}")
            take_nr = int(take_nr)
            if take_nr == 1 and take_1[id.item()] == 0:
                take_1[id.item()] = 1
                image_paths_wrong.append(im_paths[i])
                names_wrong.append(ids_and_names[id.item()])
                predicts_wrong.append(ids_and_names[preds[i].item()])
            if take_nr == 5 and take_5[id.item()] == 0:
                take_5[id.item()] = 1
                image_paths_wrong.append(im_paths[i])
                names_wrong.append(ids_and_names[id.item()])
                predicts_wrong.append(ids_and_names[preds[i].item()])


class REIDModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.params = hparams
        classifier = nn.Sequential(*[nn.Linear(2048, self.params["num_classes"])])
        self.classifier = classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) 
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        return optimizer

    def grad_cam_overlay(roles, params, imgs, ids, im_path, model):
            for j in range(params["batch_size"]):
                if roles[j] != "patient":
                    continue
                input_tensor = torch.tensor(imgs[j], requires_grad=True)
                image = grad_cam(model, imgs[j], target_layers, ids[j])
                image = image[:, :, ::-1]
                im_array = numpy.asarray(image)
                out_path = im_path[j].split("/")[-1]
                out_path = out_path.split(".")[0]
                out_path = out_path + im_path[j].split("/")[-2]
                out_path = out_path + im_path[j].split("/")[-3]
                out_path = out_path + im_path[j].split("/")[-4]
                out_path = project_path / f"{out_path}_nr{j}.jpg"
                cv2.imwrite(str(out_path), im_array)
                
    def training_step(self, batch, batch_idx):
        imgs = batch['image']
        ids = batch['id']
        im_path = batch['im_path']
        roles = batch['role']
        outputs = self.forward(imgs)
        loss = F.cross_entropy(outputs, ids)
        _, preds = torch.max(outputs, 1)
        acc = preds.eq(ids).sum().float() / ids.size(0)

        if self.current_epoch == 2:
            self.grad_cam_overlay(roles, self.params, imgs, ids, im_path, self.model)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", 100 * acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        imgs_val = batch['image']
        ids_val = batch['id']
        outputs_val = self.forward(imgs_val)
        loss_val = F.cross_entropy(outputs_val, ids_val)
        _, preds_val = torch.max(outputs_val, 1)
        acc_val = preds_val.eq(ids_val).sum().float() / ids_val.size(0)

        self.log("val_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", 100 * acc_val, prog_bar=True, on_step=False, on_epoch=True)

        return {'val_loss': loss_val, 'val_acc': acc_val, 'cf_preds': preds_val, 'cf_labels': ids_val}

    def validation_epoch_end(self, outputs):
        preds_val = torch.cat([tmp['cf_preds'] for tmp in outputs])
        targets = torch.cat([tmp['cf_labels'] for tmp in outputs])
        confusion_matrix = torchmetrics.functional.confusion_matrix(preds_val, targets, num_classes=5, normalize='true')
    
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(),
                             index=[ids_and_names[i] for i in range(self.params['num_classes'])],
                             columns=[ids_and_names[i] for i in range(self.params['num_classes'])])
    
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='.2%')
        plt.savefig(f'cf_matrix_trainwvalid_{self.current_epoch + 1}.png')


class ReidData(pl.LightningDataModule):

    def __init__(self, batch_size=4, resolution=224):
        super().__init__()
        self.batch_size = batch_size
        self.resolution = resolution
        self.reid_dataset_val = None
        self.reid_dataset_train = None

    def setup(self):

        transform = transforms.Compose([
            SquarePad2(),
            transforms.Resize((self.resolution, self.resolution)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        df_path = Path("/tmp/pycharm_project_751")
        df_train_path = df_path / 'df_train.pkl'
        df_val_path = df_path / 'df_val.pkl'

        self.reid_dataset_train = ReIDDataset(path=str(df_train_path), transform=transform)
        self.reid_dataset_val = ReIDDataset(path=str(df_val_path), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.reid_dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.reid_dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=True)


if __name__ == '__main__':
    project_path = Path("/home/anaml/gradcam_patient")
    hparams = {
        "batch_size": 16,
        "lr": 0.0002,
        "weight_decay": 5e-4,
        "nr_epochs": 7,
        "num_classes": 5,
        "resolution": 224
    }

    model = REIDModel(hparams)
    data = ReidData(hparams["batch_size"])
    data.setup()
    trainer = pl.Trainer(
        max_epochs=hparams["nr_epochs"],
        gpus=1
    )

    target_layers = model.model.layer4[-1]

    trainer.fit(model,train_dataloaders=data.train_dataloader(),val_dataloaders=data.val_dataloader())
    torch.save(model.state_dict(), 'model.pth')
    
    # with open("wrongly_classified.pkl", "wb") as f:
    #     pickle.dump(image_paths_wrong, f)
    #
    # with open("wrong_names.pkl", "wb") as f:
    #     pickle.dump(predicts_wrong, f)
    #
    # with open("actual_names.pkl", "wb") as f:
    #     pickle.dump(names_wrong, f)

    # with open("correctly_classified.pkl", "wb") as f:
    #     pickle.dump(image_paths_correct, f)
    #
    # with open("correct_names.pkl", "wb") as f:
    #     pickle.dump(names_correct, f)
