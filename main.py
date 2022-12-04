import json
import pickle
import threading
from pathlib import Path
from time import sleep

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from tqdm import tqdm

from utils import load_timestamp_infos, human_pose_to_mesh_and_joints, load_gt_role_labels, coord_transform_human_pose_tool_to_inm


if __name__ == '__main__':
    INM_DATA_ROOT_PATH = Path('/mnt/polyaxon/data1/4D-OR')
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
    TAKE_IDX = 1  # Between 1 and 10
    root_path = INM_DATA_ROOT_PATH / f'export_holistic_take{TAKE_IDX}_processed'
    timestamp_path = root_path / 'eventlog.cpb'
    pcd_path = root_path / 'pcds'
    annotations_path = root_path / 'annotations'
    timestamp_infos, lowest_value, highest_value = load_timestamp_infos(timestamp_path)
    rgb_path = root_path / 'colorimage'
    # Load the clinical roles
    take_frame_to_human_idx_to_name_and_joints = load_gt_role_labels(TAKE_IDX, INM_DATA_ROOT_PATH, TAKE_SPLIT)
    step_size = int(1000000000 / FPS)
    cameras = ["camera01", "camera02", "camera03", "camera04", "camera05", "camera06"]

    # ---------------START: The main logic for loading the human poses-------------------------------
    with (root_path / 'timestamp_to_pcd_and_frames_dict.json').open() as f:
        timestamp_to_pcd_and_frames = json.load(f)
        timestamp_to_pcd_and_frames = {int(k): v for k, v in timestamp_to_pcd_and_frames.items()}
    i = 0
    for idx, timestamp in tqdm(enumerate(range(lowest_value, highest_value, step_size)),
                               total=(highest_value - lowest_value) // step_size):
        i+=1
        # print(f"{timestamp=}")
        image_idx_str = timestamp_to_pcd_and_frames[timestamp]['color_1']
        pcd_idx_str = timestamp_to_pcd_and_frames[timestamp]['pcd']
        merged_pcd = o3d.io.read_point_cloud(str(pcd_path / f'{pcd_idx_str}.pcd'))
        # print(f'{image_idx_str=}')
        for cam in cameras:
            image_path = rgb_path / f'{cam}_colorimage-{image_idx_str}.jpg'
            # print(f'{image_path=}')
        human_pose_json_path = annotations_path / f'{pcd_idx_str}.json'
        # print(f'{human_pose_json_path=}')

        jointmesh = None
        text_positions = []
        text_labels = []

        if human_pose_json_path.exists():
            # Load the human pose annotations
            with human_pose_json_path.open() as f:
                human_pose_json = json.load(f)

            human_names = sorted({elem['humanName'] for elem in human_pose_json['labels']})
            print(f'{human_names=}')
            h_idx = 0
            for human_name in human_names:
                human_pose = []
                human_joints = [elem for elem in human_pose_json['labels'] if elem['humanName'] == human_name]
                print(f"{human_joints=}")
                joint_positions = {}
                track_idx = -1
                for human_joint in human_joints:
                    track_idx = human_joint['track_idx']
                    joint_positions[human_joint['jointName']] = (
                        human_joint['point3d']['location']['x'], human_joint['point3d']['location']['y'],
                        human_joint['point3d']['location']['z'])
                    print(f"{joint_positions=}")
                is_interpolated = all([elem['is_interpolated'] for elem in human_joints])
                for body_part in IDX_TO_BODY_PART:
                    human_pose.append(joint_positions[body_part])
                print(f"{human_pose=}")
                human_pose = np.asarray(human_pose)
                print(f"{human_pose=}")
                # Convert human pose to correct format
                human_pose = coord_transform_human_pose_tool_to_inm(human_pose)
                print(f"{human_pose=}")
                if human_name == 'Patient':
                    h_name = 'Patient'
                else:
                    h_name = f'human_{h_idx}'
                    h_idx += 1
                _, linemesh = human_pose_to_mesh_and_joints(human_pose, track_idx, LIMBS, HUMAN_POSE_COLOR_MAP,
                                                            is_interpolated=is_interpolated)
                if jointmesh is None:
                    jointmesh = linemesh
                else:
                    jointmesh += linemesh
                print(f"{jointmesh=}")
        if i == 32:
            o3d.visualization.draw_geometries([merged_pcd])
            break

    print("done")