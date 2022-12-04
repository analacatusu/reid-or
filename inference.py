from pathlib import Path
from utils import load_timestamp_infos
from tqdm import tqdm
import json
import cv2
import pickle
import os
from torchvision import transforms
from transformations import SquarePad2
from or_reid import LitModel
import torch
from PIL import Image


def bb_in_range(xmin, xmax, ymin, ymax, imx, imy):
    if xmin < 0:
        return False
    if ymin < 0:
        return False
    if xmax > imx:
        return False
    if ymax > imy:
        return False
    return True


def change_bb(xmin, xmax, ymin, ymax, imx, imy, offset_x, offset_y):
    if xmin - offset_x >= 0:
        xmin = xmin - offset_x
    if xmax + offset_x <= imx:
        xmax = xmax + offset_x
    if ymin - offset_y >= 0:
        ymin = ymin - offset_y
    if ymax + offset_y <= imy:
        ymax = ymax + offset_y
    return xmin, xmax, ymin, ymax


def func1(bb_dict):
    bounding_boxx = []
    for key, values in bb_dict.items():
        if key.startswith("human") or key.startswith("Patient"):
            if key == "Patient":
                current_role = "patient"
                min_id = values[-1]
            else:
                try:
                    current_role = track_to_role_dict[str(values[-1])]
                except KeyError:
                    continue
                track_idx_current_role = [int(k) for k,v in track_to_role_dict.items() if v == current_role]
                min_id = min(track_idx_current_role)
                values[-1] = min_id

            bounding_boxx.append(values)
    return bounding_boxx

# filename = 'finalized_model.pkl'
# model = pickle.load(open(filename, 'rb'))
hparams = {
        "batch_size": 16,
        "lr": 0.0002,
        "weight_decay": 5e-4,
        "nr_epochs": 7,
        "num_classes": 5,
        "resolution": 224
    }
model2 = torch.load('model.pt')
print("load1")
model_state = LitModel(hparams)
model_state.load_state_dict(torch.load('model.pth'))
print("load2")
model2.eval()
model_state.eval()

transform = transforms.Compose([
    SquarePad2(),
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

INM_DATA_ROOT_PATH = Path('/mnt/polyaxon/data1/4D-OR')
TAKE_IDX = 1  # Pick a take between 1 and 10

### take_5
# role_and_name = {"patient": "lennart",
#                  "head-surgeon": "chantal",
#                  "assistant-surgeon": "ege",
#                  "circulating-nurse": "evin",
#                  "anaesthetist": "tianyu"
#                  }
### take_1
role_and_name = {"patient" : "lennart",
         "head-surgeon" : "ege",
         "assistant-surgeon" : "chantal",
         "circulating-nurse" : "tianyu",
         "anaesthetist" : "evin"
         }

FPS = 1.  # Set to either 1.0 or 30. Point clouds are only shown in 1.0. If set to 30., only human poses are shown.
root_path = INM_DATA_ROOT_PATH / f'export_holistic_take{TAKE_IDX}_processed'
timestamp_path = root_path / 'eventlog.cpb'
timestamp_infos, lowest_value, highest_value = load_timestamp_infos(timestamp_path)
rgb_path = root_path / 'colorimage'
step_size = int(1000000000 / FPS)
bounding_boxes_path = root_path / 'object_bounding_boxes'
cameras = ["camera01", "camera02", "camera03", "camera04", "camera05", "camera06"]

with (root_path / 'track_to_role.json').open() as f:
    track_to_role_dict = json.load(f)

with (root_path / 'timestamp_to_pcd_and_frames_dict.json').open() as f:
    timestamp_to_pcd_and_frames = json.load(f)
    timestamp_to_pcd_and_frames = {int(k): v for k, v in timestamp_to_pcd_and_frames.items()}

i = 0
for idx, timestamp in tqdm(enumerate(range(lowest_value, highest_value, step_size)),
                           total=(highest_value - lowest_value) // step_size):
    i += 1
    if i == 767:
        continue
    try:
        pcd_idx_str = timestamp_to_pcd_and_frames[timestamp]['pcd']
    except KeyError:
        continue
    for index, cam in enumerate(cameras):
        image_idx_str = timestamp_to_pcd_and_frames[timestamp][f'color_{index + 1}']
        image_path = rgb_path / f'{cam}_colorimage-{image_idx_str}.jpg'
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        imx, imy = image.shape[:2]
        bb_path = bounding_boxes_path / f'{pcd_idx_str}_{index + 1}.json'
        bounding_boxes = []
        if bb_path.exists():
            with bb_path.open() as f:
                bb_dict = json.load(f)
                bounding_boxes = func1(bb_dict)
        for bb in bounding_boxes:
            o_xmin, o_xmax, o_ymin, o_ymax, track_idx = bb
            if not bb_in_range(o_xmin, o_xmax, o_ymin, o_ymax, imx, imy):
                continue
            xmin, xmax, ymin, ymax = change_bb(o_xmin, o_xmax, o_ymin, o_ymax, imx, imy, 20, 80)
            image_cropped = image[ymin:ymax, xmin:xmax]
            if image_cropped.size == 0:
                continue
            image_cropped = Image.fromarray(image_cropped.astype('uint8'), 'RGB')
            image_transformed = transform(image_cropped)
            image_transformed = image_transformed.unsqueeze(0)
            outputs = model2(image_transformed)  ## should be a label
            outputs = outputs.view(1, -1)
            _, label = torch.max(outputs, 1)
            print(f"{label=}")
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2) ## white box
            cv2.putText(image, f"id: {label.item()}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        output_path = Path(f"/home/anaml/inference2/take_{TAKE_IDX}/")
        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as error:
            print("Something went wrong. Should not ended up here.")
        output = output_path / f'{cam}_colorimage-{image_idx_str}.jpg'
        cv2.imwrite(str(output), image)

print("done")
