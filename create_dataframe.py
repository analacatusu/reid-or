import pickle
import numpy as np
from pathlib import Path
import pandas as pd
import os


name_to_trackidx_path = Path("/tmp/pycharm_project_751/names_and_trackidx.pkl")
with open(str(name_to_trackidx_path), "rb") as f:
    name_to_trackidx = pickle.load(f)
    print(f"{name_to_trackidx=}")
IDS_list = []
NAMES_list = []

for k,v in name_to_trackidx.items():
    IDS_list.append(v)
    NAMES_list.append(k)

"""
Construct a pandas dataframe for the dataset
"""

### take_1
role_and_name_1 = {"patient" : "lennart",
             "head-surgeon" : "ege",
             "assistant-surgeon" : "chantal",
             "circulating-nurse" : "tianyu",
             "anaesthetist" : "evin"
             }
### take_2
role_and_name_2 = {"patient":"lennart",
                     "head-surgeon":"ege",
                     "assistant-surgeon":"chantal",
                     "circulating-nurse":"tianyu",
                     "anaesthetist":"evin"
                     }

### take_3
role_and_name_3 = {"patient":"lennart",
                     "head-surgeon":"ege",
                     "assistant-surgeon":"chantal",
                     "circulating-nurse":"tianyu",
                     "anaesthetist":"evin"
                     }

### take_4
role_and_name_4 = {"patient":"lennart",
                     "head-surgeon":"chantal",
                     "assistant-surgeon":"ege",
                     "circulating-nurse":"evin",
                     "anaesthetist":"tianyu"
                     }
### take_5
role_and_name_5 = {"patient":"lennart",
                     "head-surgeon":"chantal",
                     "assistant-surgeon":"ege",
                     "circulating-nurse":"evin",
                     "anaesthetist":"tianyu"
                     }
### take_6
role_and_name_6 = {"patient":"ege",
                     "head-surgeon":"evin",
                     "assistant-surgeon":"tianyu",
                     "circulating-nurse":"chantal",
                     "anaesthetist":"lennart"
                     }
### take_7
role_and_name_7 = {"patient":"ege",
                     "head-surgeon":"evin",
                     "assistant-surgeon":"tianyu",
                     "circulating-nurse":"chantal",
                     "anaesthetist":"lennart"
                     }
### take_8
role_and_name_8 = {"patient":"ege",
                     "head-surgeon":"tianyu",
                     "assistant-surgeon":"evin",
                     "circulating-nurse":"lennart",
                     "anaesthetist":"chantal"
                     }
### take_9
role_and_name_9 = {"patient":"ege",
                     "head-surgeon":"tianyu",
                     "assistant-surgeon":"evin",
                     "circulating-nurse":"lennart",
                     "anaesthetist":"chantal"
                     }
#### take_10
role_and_name_10 = {"patient":"tianyu",
                     "head-surgeon":"lennart",
                     "assistant-surgeon":"ege",
                     "circulating-nurse":"evin",
                     "anaesthetist":"chantal"
                     }

def find_role(index, name):
    if index + 1 == 1:
        for k, v in role_and_name_1.items():
            if v == name:
                return k
    if index + 1 == 2:
        for k, v in role_and_name_2.items():
            if v == name:
                return k
    if index + 1 == 3:
        for k, v in role_and_name_3.items():
            if v == name:
                return k
    if index + 1 == 4:
        for k, v in role_and_name_4.items():
            if v == name:
                return k
    if index + 1 == 5:
        for k, v in role_and_name_5.items():
            if v == name:
                return k
    if index + 1 == 6:
        for k, v in role_and_name_6.items():
            if v == name:
                return k
    if index + 1 == 7:
        for k, v in role_and_name_7.items():
            if v == name:
                return k
    if index + 1 == 8:
        for k, v in role_and_name_8.items():
            if v == name:
                return k
    if index + 1 == 9:
        for k, v in role_and_name_9.items():
            if v == name:
                return k
    if index + 1 == 10:
        for k, v in role_and_name_10.items():
            if v == name:
                return k

def construct_pd(root_dir):
    prepare_df = []
    for (root, dirs, files) in os.walk(root_dir, topdown=True):
        dirs.sort()
        if len(files) != 0:
            for file in files:
                im_path = os.path.join(root, file)
                take_idx = root.split('/')[-3]
                index = take_idx.split('_')[-1]
                index = int(index) - 1
                string_id = root.split('/')[-1]
                string_id = string_id.split('_')[-1]
                int_id = -1
                name = ""
                role= ""
            for i, l in enumerate(IDS_list):
                if l[index] == int(string_id):
                    int_id = i
                    name = NAMES_list[i]
                    role= find_role(index, name)
                    prepare_df.append([im_path, int_id, name, role])
            	     
    return prepare_df        	     

if __name__ == '__main__':
    root_dir = Path("/home/anaml/inputs/val")
    prepare_df = construct_pd(root_dir)
    df_array = np.array(prepare_df)
    df = pd.DataFrame(df_array, columns=['im_path', 'label', 'name', 'role'])
    df.to_pickle("/tmp/pycharm_project_751/df_val.pkl")

    root_dir = Path("/home/anaml/inputs/train")
    prepare_df = construct_pd(root_dir)
    df_array = np.array(prepare_df)
    df = pd.DataFrame(df_array, columns=['im_path', 'label', 'name', 'role'])
    df.to_pickle("/tmp/pycharm_project_751/df_train.pkl")
