    
from pathlib import Path
from dataset2 import ReIDDataset
import pandas as pd
import torch
from or_reid import REIDModel

df_path = Path("/tmp/pycharm_project_751")
df_val_path = df_path / 'df_val.pkl'
hparams = {
        "batch_size": 16,
        "lr": 0.0002,
        "weight_decay": 5e-4,
        "nr_epochs": 7,
        "num_classes": 5,
        "resolution": 224
    }
model = REIDModel(hparams)
model.load_state_dict(torch.load('model.pth'))

reid_dataset = ReIDDataset(path=str(df_val_path), transform=None)

path = Path('/tmp/pycharm_project_751/df_take_1.pkl')
path2 = Path('/tmp/pycharm_project_751/df_take_5.pkl')
df1 = pd.read_pickle(str(path))
df5 = pd.read_pickle(str(path2))

for i_batch, sample in enumerate(reid_dataset):
        image = sample['image']
        image_path = sample['im_path']
        output = model(image).data.cpu().numpy()
        index1 = df1.index[df1['input_path'] == str(image_path)].tolist()
        index5 = df5.index[df5['input_path'] == str(image_path)].tolist()
        if len(index1) != 0:
            df1.iloc[index1[0]]['label'] = output

        if len(index1) != 0:
            df5.iloc[index5[0]]['label'] = output

df1.to_pickle("/tmp/pycharm_project_751/df_take_1_labeled.pkl")
df5.to_pickle("/tmp/pycharm_project_751/df_take_5_labeled.pkl")
