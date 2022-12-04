import json
import pickle
from pathlib import Path
from tqdm import tqdm
import cv2
import os
from collections import defaultdict
from utils import load_timestamp_infos, load_gt_role_labels



def bb_in_range(xmin,xmax,ymin,ymax,imx,imy):
    if xmin < 0:
        return False
    if ymin < 0:
        return False
    if xmax > imx:
        return False
    if ymax > imy:
        return False
    return True


def change_bb(xmin,xmax,ymin,ymax,imx,imy,offset_x,offset_y):
    if xmin - offset_x >= 0:
        xmin = xmin - offset_x
    if xmax + offset_x <= imx:
        xmax = xmax + offset_x
    if ymin - offset_y >= 0:
        ymin = ymin - offset_y
    if ymax + offset_y <= imy:
        ymax = ymax + offset_y
    return xmin, xmax, ymin, ymax


def _default():
    return -1


def refine_bb(bb_dict):
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

            if len(names_and_trackidx[role_and_name[current_role]]) < TAKE_IDX:
                names_and_trackidx[role_and_name[current_role]][TAKE_IDX - 1] = min_id
            bounding_boxx.append(values)
    return bounding_boxx


if __name__ == '__main__':
    INM_DATA_ROOT_PATH = Path('/mnt/polyaxon/data1/4D-OR')
    TAKE_IDX = 5  # Pick a take between 1 and 10
    ### take_1
    # role_and_name = {"patient" : "lennart",
    #          "head-surgeon" : "ege",
    #          "assistant-surgeon" : "chantal",
    #          "circulating-nurse" : "tianyu",
    #          "anaesthetist" : "evin"
    #          }
    ### take_2
    # role_and_name = {"patient":"lennart",
    #                  "head-surgeon":"ege",
    #                  "assistant-surgeon":"chantal",
    #                  "circulating-nurse":"tianyu",
    #                  "anaesthetist":"evin"
    #                  }

    ### take_3
    # role_and_name = {"patient":"lennart",
    #                  "head-surgeon":"ege",
    #                  "assistant-surgeon":"chantal",
    #                  "circulating-nurse":"tianyu",
    #                  "anaesthetist":"evin"
    #                  }

    ### take_4
    # role_and_name = {"patient":"lennart",
    #                  "head-surgeon":"chantal",
    #                  "assistant-surgeon":"ege",
    #                  "circulating-nurse":"evin",
    #                  "anaesthetist":"tianyu"
    #                  }
    ### take_5
    role_and_name = {"patient":"lennart",
                     "head-surgeon":"chantal",
                     "assistant-surgeon":"ege",
                     "circulating-nurse":"evin",
                     "anaesthetist":"tianyu"
                     }
    ### take_6
    # role_and_name = {"patient":"ege",
    #                  "head-surgeon":"evin",
    #                  "assistant-surgeon":"tianyu",
    #                  "circulating-nurse":"chantal",
    #                  "anaesthetist":"lennart"
    #                  }
    ### take_7
    # role_and_name = {"patient":"ege",
    #                  "head-surgeon":"evin",
    #                  "assistant-surgeon":"tianyu",
    #                  "circulating-nurse":"chantal",
    #                  "anaesthetist":"lennart"
    #                  }
    ### take_8
    # role_and_name = {"patient":"ege",
    #                  "head-surgeon":"tianyu",
    #                  "assistant-surgeon":"evin",
    #                  "circulating-nurse":"lennart",
    #                  "anaesthetist":"chantal"
    #                  }
    ### take_9
    # role_and_name = {"patient":"ege",
    #                  "head-surgeon":"tianyu",
    #                  "assistant-surgeon":"evin",
    #                  "circulating-nurse":"lennart",
    #                  "anaesthetist":"chantal"
    #                  }
    # ### take_10
    # role_and_name = {"patient":"tianyu",
    #                  "head-surgeon":"lennart",
    #                  "assistant-surgeon":"ege",
    #                  "circulating-nurse":"evin",
    #                  "anaesthetist":"chantal"
    #                  }

    if TAKE_IDX == 1:
        names_and_trackidx = defaultdict(list)
    else:
        with open('names_and_trackidx.pkl', 'rb') as f:
            names_and_trackidx = pickle.load(f)
    
    print(f"{names_and_trackidx=}")
    TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}
    IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow',
                        'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                        'rightknee', 'leftfoot', 'rightfoot']
    LIMBS = [
        [5, 4],  # (righthip-lefthip)
        [9, 7],  # (rightwrist - rightelbow)
        [7, 3],  # (rightelbow - rightshoulder)
        [2, 6],  # (leftshoulder - leftelbow)
        [6, 8],  # (leftelbow - leftwrist)
        [5, 3],  # (righthip - rightshoulder)
        [4, 2],  # (lefthip - leftshoulder)
        [3, 1],  # (rightshoulder - neck)
        [2, 1],  # (leftshoulder - neck)
        [1, 0],  # (neck - head)
        [10, 4],  # (leftknee,lefthip),
        [11, 5],  # (rightknee,righthip),
        [12, 10],  # (leftfoot,leftknee),
        [13, 11]  # (rightfoot,rightknee),

    ]

    HUMAN_POSE_COLOR_MAP = {
        0: (155, 0, 0),  # red
        1: (0, 255, 0),  # green
        2: (0, 0, 255),  # blue
        3: (255, 150, 0),  # orange
        4: (0, 255, 255),  # turquise
        5: (150, 0, 255),  # purple
        6: (255, 255, 0),  # yellow
        7: (255, 0, 255),  # pink
    }
    FPS = 1.  # Set to either 1.0 or 30. Point clouds are only shown in 1.0. If set to 30., only human poses are shown.
    root_path = INM_DATA_ROOT_PATH / f'export_holistic_take{TAKE_IDX}_processed'
    timestamp_path = root_path / 'eventlog.cpb'
    pcd_path = root_path / 'pcds'
    annotations_path = root_path / 'annotations'
    timestamp_infos, lowest_value, highest_value = load_timestamp_infos(timestamp_path)
    rgb_path = root_path / 'colorimage'
    # Load the clinical roles
    take_frame_to_human_idx_to_name_and_joints = load_gt_role_labels(TAKE_IDX, INM_DATA_ROOT_PATH, TAKE_SPLIT)
    step_size = int(1000000000 / FPS)
    keypoint_annotations_2D_path = root_path / '2D_keypoint_annotations.json'
    bounding_boxes_path = root_path / 'object_bounding_boxes'
    cameras = ["camera01", "camera02", "camera03", "camera04", "camera05", "camera06"]

    # ---------------START: The main logic for loading the human poses-------------------------------
    with (root_path / 'track_to_role.json').open() as f:
        track_to_role_dict = json.load(f)

    with (root_path / 'timestamp_to_pcd_and_frames_dict.json').open() as f:
        timestamp_to_pcd_and_frames = json.load(f)
        timestamp_to_pcd_and_frames = {int(k): v for k, v in timestamp_to_pcd_and_frames.items()}

    with (keypoint_annotations_2D_path).open() as f:
        keypoints_2D = json.load(f)

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
            imx,imy = image.shape[:2]
            bb_path = bounding_boxes_path / f'{pcd_idx_str}_{index + 1}.json'
            bounding_boxes = []
            if bb_path.exists():
                with bb_path.open() as f:
                    bb_dict = json.load(f)
                    bounding_boxes = refine_bb(bb_dict)
            for bb in bounding_boxes:
                o_xmin, o_xmax, o_ymin, o_ymax, track_idx = bb
                if not bb_in_range(o_xmin,o_xmax,o_ymin,o_ymax,imx,imy):
                    continue
                xmin, xmax, ymin, ymax = change_bb(o_xmin,o_xmax,o_ymin,o_ymax,imx,imy,20,80)

                
                image2 = image[ymin:ymax, xmin:xmax]
                if image2.size == 0:
                    continue
               
                output_path = ""
                if TAKE_IDX == 1 or TAKE_IDX == 5:
                    output_path = Path(f"/home/anaml/inputs/val/take_{TAKE_IDX}/{cam}/id_{track_idx}")
                else:
                    output_path = Path(f"/home/anaml/inputs/train/take_{TAKE_IDX}/{cam}/id_{track_idx}")

                try:
                    os.makedirs(output_path, exist_ok=True)
                except OSError as error:
                    print("Something went wrong. Should not ended up here.")

              
    with open("names_and_trackidx.pkl", "wb") as f:
        pickle.dump(names_and_trackidx, f)

