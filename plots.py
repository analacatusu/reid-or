import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
import pickle


root_path = Path("/home/anaml/inputs")
root_path2 = Path("/tmp/pycharm_project_751")

correctly_classified_path = Path("/tmp/pycharm_project_751/correctly_classified.pkl")
wrongly_classified_path = Path("/tmp/pycharm_project_751/wrongly_classified.pkl")
correct_names_path = Path("/tmp/pycharm_project_751/correct_names.pkl")
wrong_names_path = Path("/tmp/pycharm_project_751/wrong_names.pkl")
actual_names_path = Path("/tmp/pycharm_project_751/actual_names.pkl")

with open(str(correctly_classified_path), "rb") as f:
    correctly_classified = pickle.load(f)
    print(f"{correctly_classified=}")

with open(str(correct_names_path), "rb") as f:
    correct_names = pickle.load(f)
    print(f"{correct_names=}")

with open(str(wrongly_classified_path), "rb") as f:
    wrongly_classified = pickle.load(f)
    print(f"{wrongly_classified=}")

with open(str(wrong_names_path), "rb") as f:
    wrong_names = pickle.load(f)
    print(f"{wrong_names=}")

with open(str(actual_names_path), "rb") as f:
    actual_names = pickle.load(f)
    print(f"{actual_names=}")

# create figure
fig = plt.figure(figsize=(24, 24))

# setting values to rows and column variables
rows = 2
columns = 5
index = 1


for i, path in enumerate(correctly_classified):


    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fig.add_subplot(rows, columns, index)
    index += 1
    take_nr = path.split('/')[-4]
    print(f"{take_nr=}")
    take_nr = take_nr.split('_')[-1]
    print(f"{take_nr=}")
    take_nr = int(take_nr)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"take_{take_nr}_{correct_names[i]}")

plt.savefig("correctly_classified.png")

fig = plt.figure(figsize=(24, 24))
index = 1

for i, path in enumerate(wrongly_classified):


    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fig.add_subplot(rows, columns, index)
    index += 1
    take_nr = path.split('/')[-4]
    print(f"{take_nr=}")
    take_nr = take_nr.split('_')[-1]
    print(f"{take_nr=}")
    take_nr = int(take_nr)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"take_{take_nr}_pred:{wrong_names[i]}_act:{actual_names[i]}")

plt.savefig("wrongly_classified.png")

train_path = root_path / "train"
val_path = root_path / "val"
datasets = {}
reid_dataset_train = ReIDDataset(root_dir=str(train_path))
reid_dataset_val = ReIDDataset(root_dir=str(val_path))

name_to_trackidx_path = Path("/tmp/pycharm_project_751/names_and_trackidx.pkl")
with open(str(name_to_trackidx_path), "rb") as f:
    name_to_trackidx = pickle.load(f)
    print(f"{name_to_trackidx=}")

def search(index, id):
    if name_to_trackidx['lennart'][index] == id:
        return 'Lennart'
    if name_to_trackidx['evin'][index] == id:
        return 'Evin'
    if name_to_trackidx['tianyu'][index] == id:
        return 'Tianyu'
    if name_to_trackidx['ege'][index] == id:
        return 'Ege'
    if name_to_trackidx['chantal'][index] == id:
        return 'Chantal'

# create figure
fig = plt.figure(figsize=(24, 24))

# setting values to rows and column variables
rows = 10
columns = 5
index = 1


for (root, dirs, files) in os.walk(root_path, topdown=True):
    dirs.sort()
    # print(f"{root=}")
    # print(f"{dirs=}")
    # print(f"{files=}")
    if len(files) != 0:
        print(root)
        camera = root.split('/')[-2]
        if camera != 'camera01':
            continue
        id = root.split('/')[-1]
        id = id.split('_')[-1]
        id = int(id)
        take_nr = root.split('/')[-3]
        take_nr = take_nr.split('_')[-1]
        take_nr = int(take_nr)
        name = search(take_nr - 1, id)
        print(f"{files[0]=}")
        im_path = os.path.join(root, files[0])
        image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fig.add_subplot(rows, columns, index)
        index += 1

        plt.imshow(image)
        plt.axis('off')
        plt.title(f"take_{take_nr}_{name}")

plt.savefig("table.png")

for i, data in enumerate(reid_dataset_train):
     img = data['image']
     id = data['id']
     path = data['im_path']
     print(f"{path=}")
     take_nr = path.split('/')[-4]
     print(f"{take_nr=}")
     take_nr = take_nr.split('_')[-1]
     take_nr_int = int(take_nr)
     if takes[take_nr_int - 1]:
         continue
     takes[take_nr_int - 1] = True
     fig.add_subplot(rows, columns, index)
     index += 1

     plt.imshow(img)
     plt.axis('off')
     plt.title(f"take_{take_nr}_{ids_and_names[id]}")


for i, data in enumerate(reid_dataset_val):
     img = data['image']
     id = data['id']
     path = data['im_path']
     take_nr = path.split('/')[-4]
     print(f"{take_nr=}")
     take_nr = take_nr.split('_')[-1]
     take_nr_int = int(take_nr - 1)
     if takes[take_nr_int - 1]:
         continue
     takes[take_nr_int] = True
     fig.add_subplot(rows, columns, index)
     index += 1

     plt.imshow(img)
     plt.axis('off')
     plt.title(f"take_{take_nr}_{ids_and_names[id]}")
