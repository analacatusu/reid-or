import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                if o3d.__version__ == '0.9.0.0':
                    cylinder_segment = cylinder_segment.rotate(
                        R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                else:  # assume newer or equal 0.10.0.0
                    cylinder_segment = cylinder_segment.rotate(
                        R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def kinect_to_inm_pose(kinect_pose):
    '''
    Kinect Pose Format:
    0:PELVIS,1:SPINE_NAVAL,2:SPINE_CHEST,3:NECK,4:CLAVICLE_LEFT,5:SHOULDER_LEFT,6:ELBOW_LEFT,7:WRIST_LEFT,
    8:HAND_LEFT,9:HANDTIP_LEFT,10:THUMB_LEFT,11:CLAVICLE_RIGHT,12:SHOULDER_RIGHT,13:ELBOW_RIGHT,14:WRIST_RIGHT,
    15:HAND_RIGHT,16:HANDTIP_RIGHT,17:THUMB_RIGHT,18:HIP_LEFT,19:KNEE_LEFT,20:ANKLE_LEFT,21:FOOT_LEFT,22:HIP_RIGHT,
    23:KNEE_RIGHT,24:ANKLE_RIGHT,25:FOOT_RIGHT,26:HEAD,27:NOSE,28:EYE_LEFT,29:EAR_LEFT,30:EYE_RIGHT,31:EAR_RIGHT,
    INM Pose Format:
    0:HEAD,1:NECK,2:SHOULDER_LEFT,3:SHOULDER_RIGHT,4:HIP_LEFT,5:HIP_RIGHT,6:ELBOW_LEFT,7:ELBOW_RIGHT,8:WRIST_LEFT,9:WRIST_RIGHT,10: KNEE_LEFT, 11: KNEE_RIGHT, 12: FOOT_LEFT, 13: FOOT_RIGHT
    :param kinect_pose: pose in kinect format
    :return: inm_pose: pose in inm format
    '''
    inm_pose = np.asarray(kinect_pose)[[26, 3, 5, 12, 18, 22, 6, 13, 7, 14, 19, 23, 21, 25]].tolist()
    return inm_pose


def human_pose_to_mesh_and_joints(human_pose, track_idx, limbs, HUMAN_POSE_COLOR_MAP, is_interpolated=False):
    points = np.asarray(human_pose) * 500
    color = [elem / 255 for elem in HUMAN_POSE_COLOR_MAP[track_idx % 8]]
    if is_interpolated:
        color = [0.5, 0.5, 0.5]
    linemeshes = []
    joint_points = o3d.utility.Vector3dVector(points)
    joint_colors = [color for _ in range(len(limbs))]
    joint_lines = o3d.utility.Vector2iVector(limbs)
    line_mesh = LineMesh(joint_points, joint_lines, joint_colors, radius=10)
    line_mesh_geom = line_mesh.cylinder_segments
    linemeshes.extend(line_mesh_geom)

    merged_line_meshes = linemeshes[0]
    for elem in linemeshes[1:]:
        merged_line_meshes += elem
    points = np.stack(points)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector([color for _ in range(len(human_pose))])
    return pc, merged_line_meshes


def get_rels_path(USE_IMAGES, VIS_CONFIG, TAKE_SPLIT):
    if VIS_CONFIG['TAKE_IDX'] in TAKE_SPLIT['train']:
        return Path('scan_relations_training_no_gt_train_scans.json') if not USE_IMAGES else Path('scan_relations_training_no_gt_images_train_scans.json')
    elif VIS_CONFIG['TAKE_IDX'] in TAKE_SPLIT['val']:
        return Path('scan_relations_training_no_gt_validation_scans.json') if not USE_IMAGES else Path(
            'scan_relations_training_no_gt_images_validation_scans.json')
    elif VIS_CONFIG['TAKE_IDX'] in TAKE_SPLIT['test']:
        return Path('scan_relations_training_no_gt_test_scans.json') if not USE_IMAGES else Path('scan_relations_training_no_gt_images_test_scans.json')

    return None

def load_cam_infos(root_path: Path, cam_count=6):
    cam_infos = {}
    for c_idx in range(1, cam_count + 1):
        cam_json_path = root_path / f'camera0{c_idx}.json'
        with cam_json_path.open() as f:
            cam_info = json.load(f)['value0']
            intrinsics_json = cam_info['color_parameters']['intrinsics_matrix']
            intrinsics = np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                                     [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                                     [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

            extrinsics_json = cam_info['camera_pose']
            trans = extrinsics_json['translation']
            rot = extrinsics_json['rotation']
            extrinsics = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            extrinsics[:3, :3] = rot_matrix
            extrinsics[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]

            color2depth_json = cam_info['color2depth_transform']
            trans = color2depth_json['translation']
            rot = color2depth_json['rotation']
            color2depth_transform = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            color2depth_transform[:3, :3] = rot_matrix
            color2depth_transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
            depth_extrinsics = np.copy(extrinsics)
            extrinsics = np.matmul(extrinsics, color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

            fov_x = cam_info['color_parameters']['fov_x']
            fov_y = cam_info['color_parameters']['fov_y']
            c_x = cam_info['color_parameters']['c_x']
            c_y = cam_info['color_parameters']['c_y']
            width = cam_info['color_parameters']['width']
            height = cam_info['color_parameters']['height']

            params = cam_info['color_parameters']['radial_distortion']
            radial_params = params['m00'], params['m10'], params['m20'], params['m30'], params['m40'], params['m50']
            params = cam_info['color_parameters']['tangential_distortion']
            tangential_params = params['m00'], params['m10']

            cam_infos[f'camera0{c_idx}'] = {'intrinsics': intrinsics, 'extrinsics': extrinsics, 'fov_x': fov_x, 'fov_y': fov_y,
                                            'c_x': c_x, 'c_y': c_y, 'width': width, 'height': height, 'radial_params': radial_params,
                                            'tangential_params': tangential_params, 'depth_extrinsics': depth_extrinsics}

    return cam_infos



def load_human_pose(predicted_pose_path, human_to_role, info, VIS_CONFIG, LIMBS, HUMAN_POSE_COLOR_MAP):
    if not VIS_CONFIG['HUMAN_POSE']:
        return
    if predicted_pose_path.exists():
        pred = np.load(str(predicted_pose_path))
        for idx, p in enumerate(pred):
            _, linemesh = human_pose_to_mesh_and_joints(p, 0, LIMBS, HUMAN_POSE_COLOR_MAP)
            if info['jointmesh'] is None:
                info['jointmesh'] = linemesh
            else:
                info['jointmesh'] += linemesh
            label_pos = p[0]  # 0 is head
            label_pos[1] += 50
            info['text_positions'].append(label_pos)
            info['text_labels'].append(human_to_role[f'human_{idx}'].replace('_', ' '))


def load_gt_role_labels(take_idx, INM_DATA_ROOT_PATH, TAKE_SPLIT):
    take_frame_to_human_idx_to_name_and_joints = {}
    root_path = INM_DATA_ROOT_PATH / 'human_name_to_3D_joints'
    rel_data_path = INM_DATA_ROOT_PATH / 'relationship_data'
    GT_take_human_name_to_3D_joints = np.load(str(root_path / f'{take_idx}_GT_True.npz'), allow_pickle=True)['arr_0'].item()
    if take_idx in TAKE_SPLIT['train']:
        gt_rels_path = rel_data_path / 'relationships_train.json'
    elif take_idx in TAKE_SPLIT['val']:
        gt_rels_path = rel_data_path / 'relationships_validation.json'
    elif take_idx in TAKE_SPLIT['test']:
        gt_rels_path = rel_data_path / 'relationships_test.json'
    else:
        raise Exception()
    with open(gt_rels_path) as f:
        all_scans_gt_rels = json.load(f)['scans']
    for scan_gt_rel in all_scans_gt_rels:
        if scan_gt_rel['take_idx'] != take_idx:
            continue
        if 'Patient' in scan_gt_rel['objects'].values():
            scan_gt_rel['human_idx_to_name']['Patient'] = 'Patient'
        take_frame_str = f'{take_idx}_{scan_gt_rel["scan"]}'
        human_indices = list(scan_gt_rel['human_idx_to_name'].keys())
        human_idx_to_human_name_and_joints = {}
        for human_idx in human_indices:
            try:
                name = scan_gt_rel['human_idx_to_name'][human_idx]
                joints = GT_take_human_name_to_3D_joints[scan_gt_rel["scan"]][human_idx]
                human_idx_to_human_name_and_joints[human_idx] = (name, joints)
            except Exception as e:
                continue

        take_frame_to_human_idx_to_name_and_joints[take_frame_str] = human_idx_to_human_name_and_joints

    return take_frame_to_human_idx_to_name_and_joints


def load_object_bbox(take_idx, scan_name, info, GROUP_FREE_PREDICTIONS_PATH, OBJECT_COLOR_MAP, OBJECT_LABEL_MAP):
    save_name = GROUP_FREE_PREDICTIONS_PATH / f'{take_idx}_{scan_name}_GT.npz'
    gt = np.load(str(save_name), allow_pickle=True)['arr_0'].item()
    classes = gt['classes']
    preds = gt['bboxes']
    preds[:, :6] *= 1000
    classes = np.asarray(classes)
    preds = np.asarray(preds)
    # optionally filter out more here
    class_color_map = {label: OBJECT_COLOR_MAP[object_name] for object_name, label in OBJECT_LABEL_MAP.items()}
    colors = [class_color_map[cls] for cls in classes]
    colors = [tuple([c for c in color]) for color in colors]

    for bbox, filtered_class, color in zip(preds, classes, colors):
        name = {v: k for k, v in OBJECT_LABEL_MAP.items()}[filtered_class]
        x_center, y_center, z_center, x_length, y_length, z_length, ang = bbox
        xmin = x_center - (x_length / 2)
        xmax = x_center + (x_length / 2)
        ymin = y_center - (y_length / 2)
        ymax = y_center + (y_length / 2)
        zmin = z_center - (z_length / 2)
        zmax = z_center + (z_length / 2)

        # xmin, xmax, ymin, ymax, zmin, zmax,_ = bbox
        bbox_points = np.asarray([(xmin, ymin, zmin), (xmin, ymin, zmax), (xmin, ymax, zmin), (xmin, ymax, zmax),
                                  (xmax, ymin, zmin), (xmax, ymin, zmax), (xmax, ymax, zmin), (xmax, ymax, zmax)])

        bbox_points = o3d.utility.Vector3dVector(bbox_points)
        bbox_lines = o3d.utility.Vector2iVector([[0, 1], [0, 4], [0, 2], [1, 5], [1, 3], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
        joint_colors = [color for _ in range(len(bbox_lines))]
        line_mesh = LineMesh(bbox_points, bbox_lines, joint_colors, radius=10)
        line_mesh_geom = line_mesh.cylinder_segments
        bbox_mesh = line_mesh_geom[0]
        for elem in line_mesh_geom[1:]:
            bbox_mesh += elem
        ang = -ang if name in (
            'operating_table', 'anesthesia_equipment') else ang  # Eventhough -ang makes more sense, for most objects ang works better
        R = bbox_mesh.get_rotation_matrix_from_xyz((0, ang, 0))
        bbox_mesh.rotate(R)
        if info['bboxmesh'] is None:
            info['bboxmesh'] = bbox_mesh
        else:
            info['bboxmesh'] += bbox_mesh


def timestamp_data_exists(timestamp_path, ch, fr):
    body_channels = [0, 3, 6, 9, 12, 15]
    color_channels = [1, 4, 7, 10, 13, 16]
    depth_channels = [2, 5, 8, 11, 14, 17]

    if ch in body_channels:
        file_name = f'bodytracking/camera0{body_channels.index(ch) + 1}_bodyjoints-{str(fr).zfill(6)}.cpb'
    elif ch in color_channels:
        file_name = f'colorimage/camera0{color_channels.index(ch) + 1}_colorimage-{str(fr).zfill(6)}.jpg'
    elif ch in depth_channels:
        file_name = f'depthimage_aligned/camera0{depth_channels.index(ch) + 1}_depthimage-{str(fr).zfill(6)}.tiff'
    else:
        return True

    data_path: Path = timestamp_path.parent / file_name
    return data_path.exists()


def load_timestamp_infos(timestamp_path, resort=False, verify=False):
    channels = [OrderedDict() for _ in range(19)]
    lowest_value = 100000000000000
    highest_value = -1
    with timestamp_path.open() as f:
        timestamp_info = json.load(f)  # For this to work, i had to manually change the eventlog file because it was not a valid json
        for elem in timestamp_info:
            elem_value = elem['value0']
            ch = elem_value['ch']
            if verify and not timestamp_data_exists(timestamp_path, ch, elem_value['fr']):
                continue
            lowest_value = min(lowest_value, elem_value['ts'])
            highest_value = max(highest_value, elem_value['ts'])
            channels[ch][elem_value['ts']] = elem_value['fr']

        if resort:
            # just to make sure they are sorted
            for idx, channel in enumerate(channels):
                channels[idx] = OrderedDict(sorted(channel.items(), key=lambda x: x[0]))

    return channels, lowest_value, highest_value


def get_closest_timestamp_in_fused_poses(timestamp_to_human_pose, wanted_timestamp, tolerance=1.0):
    key_list = list(timestamp_to_human_pose.keys())
    insert_idx = bisect.bisect_left(key_list, wanted_timestamp)
    if insert_idx >= len(key_list):
        matching_timestamp = key_list[insert_idx - 1]
    else:
        prev_value = key_list[insert_idx - 1]
        next_value = key_list[insert_idx]
        prev_distance = abs(wanted_timestamp - prev_value)
        next_distance = abs(wanted_timestamp - next_value)
        if prev_distance < next_distance:
            matching_timestamp = key_list[insert_idx - 1]
        else:
            matching_timestamp = key_list[insert_idx]

    if abs(matching_timestamp - wanted_timestamp) / 1000000000 > tolerance:
        return None
    else:
        return matching_timestamp


def coord_transform_human_pose_tool_to_inm(arr):
    # reverse of coord_transform_inm_to_human_pose_tool
    arr *= 25

    arr[:, 2] += 1000

    arr[:, 1] *= -1

    arr = arr[:, [0, 2, 1]]

    return arr
