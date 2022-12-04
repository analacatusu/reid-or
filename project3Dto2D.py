import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
from tqdm import tqdm

# from helpers.inm_helpers.configurations import IDX_TO_BODY_PART, OBJECT_POSES_PATH, OBJECT_COLOR_MAP, INM_DATA_ROOT_PATH, STATIONARY_OBJECTS
# from helpers.inm_helpers.object_pose_utils import get_object_poses
from utils import load_cam_infos, coord_transform_human_pose_tool_to_inm


def main():
    INM_DATA_ROOT_PATH = Path('/mnt/polyaxon/data1/4D-OR')
    OBJECT_POSES_PATH = Path('/mnt/polyaxon/data1/4D-OR/object_pose_results')
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
    TAKE_IDX = 1
    print(TAKE_IDX)
    root_path = INM_DATA_ROOT_PATH / f'export_holistic_take{TAKE_IDX}_processed'
    color_image_path = root_path / 'colorimage'
    annotations_path = root_path / 'annotations'
    object_poses_path = OBJECT_POSES_PATH / 'vs_0.01_rf_0.25_maxnn_500_ft_0.25'
    export_objects_bounding_boxes_path = root_path / 'object_bounding_boxes'
    export_2D_keypoint_annotations_path = root_path / '2D_keypoint_annotations.json'
    if not export_objects_bounding_boxes_path.exists():
        export_objects_bounding_boxes_path.mkdir()

    with (root_path / 'timestamp_to_pcd_and_frames_list.json').open() as f:
        timestamp_to_pcd_and_frames_list = json.load(f)

    cam_infos = load_cam_infos(root_path)
    all_keypoint_annotations_2D = {}
    person_count = 0
    for timestamp, elem in tqdm(timestamp_to_pcd_and_frames_list):
        pcd_idx_str = elem['pcd']
        human_pose_json_path = annotations_path / f'{pcd_idx_str}.json'
        if human_pose_json_path.exists():
            with human_pose_json_path.open() as f:
                human_pose_json = json.load(f)
        else:
            human_pose_json = None

        pc_objects_path = object_poses_path / f'{TAKE_IDX}_{pcd_idx_str}.npz'
        if pc_objects_path.exists():
            stationary_objects_path = object_poses_path / f'{TAKE_IDX}_stationary_objects.npz'
            json_path = object_poses_path / f'{TAKE_IDX}_{pcd_idx_str}_manual.json'
            registered_objects = np.load(str(pc_objects_path), allow_pickle=True)['arr_0'].item()
            stationary_objects = {k: v for k, v in np.load(str(stationary_objects_path), allow_pickle=True)['arr_0']}
            if pcd_idx_str > '000198' and int(TAKE_IDX) == 10:
                stationary_objects['datasets/INM/object_scans/secondary_table/10.ply'][:3, 3] += [-0.05, 0, -0.05]
            registered_objects = {k: v for k, v in registered_objects.items() if k.split("/")[3] not in STATIONARY_OBJECTS}
            registered_objects = {**registered_objects, **stationary_objects}  # merge dicts

            with json_path.open() as f:
                corresponding_json = json.load(f)
            object_poses, object_names = get_object_poses(registered_objects)

            new_object_poses = []
            new_object_name = []
            for object_pose, object_name in zip(object_poses, object_names):
                if object_name in corresponding_json['false_objects']:
                    continue
                object_pose.points = o3d.utility.Vector3dVector(np.asarray(object_pose.points) / 500)
                new_object_poses.append(object_pose)
                new_object_name.append(object_name)

            object_poses = new_object_poses
            object_names = new_object_name
        else:
            object_poses, object_names = [], []

        all_objects = []
        human_name_to_track_idx = {}

        if human_pose_json is not None:
            human_names = sorted({elem['humanName'] for elem in human_pose_json['labels']})
            h_idx = 0
            for human_name in human_names:
                human_pose = []
                human_joints = [elem for elem in human_pose_json['labels'] if elem['humanName'] == human_name]
                joint_positions = {}
                for human_joint in human_joints:
                    joint_positions[human_joint['jointName']] = (
                        human_joint['point3d']['location']['x'], human_joint['point3d']['location']['y'], human_joint['point3d']['location']['z'])

                for body_part in IDX_TO_BODY_PART:
                    human_pose.append(joint_positions[body_part])
                human_pose = np.asarray(human_pose)
                human_pose = coord_transform_human_pose_tool_to_inm(human_pose)
                human_pose /= 500
                human_pose_pc = o3d.geometry.PointCloud()
                human_pose_pc.points = o3d.utility.Vector3dVector(human_pose)
                if human_name == 'Patient':
                    h_name = 'Patient'
                else:
                    h_name = f'human_{h_idx}'
                    h_idx += 1
                human_name_to_track_idx[h_name] = human_joints[0]['track_idx']
                all_objects.append((h_name, human_pose_pc))

        for object_name, object_pose in zip(object_names, object_poses):
            all_objects.append((object_name, object_pose))

        for c_idx in range(1, 7):
            cam_info = cam_infos[f'camera0{c_idx}']
            color_image_idx_str = elem[f'color_{c_idx}']
            rgb_path = color_image_path / f'camera0{c_idx}_colorimage-{color_image_idx_str}.jpg'
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            x_bound, y_bound = rgb.shape[1], rgb.shape[0]
            object_bounding_box_path = export_objects_bounding_boxes_path / str(f'{pcd_idx_str}_{c_idx}.json')
            obj_bounding_boxes = {}
            keypoint_annotations_2D = []
            for name, obj_tmp in all_objects:
                obj = deepcopy(obj_tmp)
                obj.transform(np.linalg.inv(cam_info['extrinsics']))  # Bring from world to rgb camera coords
                obj.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # this is needed
                obj_points = np.asarray(obj.points)
                # Project onto image
                obj_points[:, 2][obj_points[:, 2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
                x = obj_points[:, 0]
                y = obj_points[:, 1]
                z = obj_points[:, 2]
                u = (x * cam_info['fov_x'] / z) + cam_info['c_x']
                v = (y * cam_info['fov_y'] / z) + cam_info['c_y']
                xmin, xmax, ymin, ymax = int(u.min()), int(u.max()), int(v.min()), int(v.max())
                if 'human' in name or 'Patient' in name:
                    color = (1, 0, 0)
                    ymin -= 10
                    ymax += 10
                    # only add object if more than half is visible in the image
                    joints_outside_image = np.logical_or(np.logical_or(u < 0, u > x_bound), np.logical_or(v < 0, v > y_bound))
                    perc_joints_outside_image = joints_outside_image.sum() / len(u)
                    if perc_joints_outside_image < 0.5:
                        keypoints = np.stack([u, v, np.ones(len(IDX_TO_BODY_PART)) + 1]).transpose().flatten().tolist()  # 2's indicate object is visible
                        keypoint_annotations_2D.append({'keypoints': keypoints, 'id': person_count})
                        person_count += 1
                    # plt.scatter(u, v, c='orange', s=10)
                elif name in OBJECT_COLOR_MAP:
                    color = OBJECT_COLOR_MAP[name]
                else:
                    color = (0, 1, 0)

                track_idx = human_name_to_track_idx.get(name, -1)
                obj_bounding_boxes[name] = (xmin, xmax, ymin, ymax, track_idx)

                # cv2.rectangle(rgb, (xmin, ymin), (xmax, ymax), tuple([int(c * 255) for c in list(color)]), thickness=1, lineType=cv2.LINE_AA)

            # currently exportet with ground truth human poses
            with object_bounding_box_path.open('w') as f:
                json.dump(obj_bounding_boxes, f)

            all_keypoint_annotations_2D[f'{pcd_idx_str}_{c_idx}'] = keypoint_annotations_2D
            # plt.imshow(rgb)
            # plt.show()

    with export_2D_keypoint_annotations_path.open('w') as f:
        json.dump(all_keypoint_annotations_2D, f)


if __name__ == '__main__':
    main()
