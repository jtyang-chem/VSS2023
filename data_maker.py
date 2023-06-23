import numpy as np
import os, sys
import json
import pandas as pd
import math
import MDAnalysis as mda

def split_np(arr, n_per_chunk):
    """ split numpy array with n_per_chunk """
    N = len(arr)
    assert N > n_per_chunk, "split numpy array error: n_per_chunk is too big"
    n_split = N//n_per_chunk 
    n_end = n_split * n_per_chunk
    res = np.split(arr[:n_end], n_split)
    res.append(arr[n_end:])
    return res

def mk_dataset(fjson, prefix):
    with open(fjson, "r") as fdat:
        dat = json.load(fdat)
    print(json.dumps(dat, indent=4, sort_keys=True))

    # read all ener and get not NaN index
    ener = np.load(dat["shift_file"]).reshape(-1)
    n_points = len(ener)
    print(f"{n_points} points in total")
    idx_not_nan = np.argwhere(~np.isnan(ener)).reshape(-1)
    n_real_points = len(idx_not_nan)
    print(f"{n_real_points} not nan points in total")
    print(idx_not_nan)

    # divide into 80% train and 20% validation randomly
    idx_not_nan_shuffle = idx_not_nan.copy()
    np.random.shuffle(idx_not_nan_shuffle) # in-place shuffle
    idx_test  = idx_not_nan_shuffle[:n_real_points//5]
    idx_train = idx_not_nan_shuffle[n_real_points//5:]

    # set path
    os.makedirs(dat["prefix"], exist_ok = True)

    val_dirs = mk_sub_data(dat, idx_test, "val")
    trn_dirs = mk_sub_data(dat, idx_train, "trn")

def mk_coord(dat, idx, dir_list):
    # read traj
    if len(dat["traj_file"]) == 1:
        traj = mda.Universe(dat["traj_file"][0])
    else:
        traj = mda.Universe(dat["traj_file"][0], dat["traj_file"][1])

    n_atoms = len(traj.atoms)
    n_dir = len(dir_list)
    for frames, d in zip(idx, dir_list):
        coords = np.empty((len(frames), n_atoms* 3))
        # loop in traj
        i = 0
        for ts in traj.trajectory[frames]:
            coords[i,  :] = traj.atoms.positions.reshape(-1)
            i+=1
        np.save(f"{d}/set.000/coord.npy", coords)

def mk_energy(dat, idx, dir_list):
    ener = np.load(dat["shift_file"]).reshape((-1))
    for frames, d in zip(idx, dir_list):
        np.save(f"{d}/set.000/energy.npy", ener[frames])

def mk_box(dat, idx, dir_list):
    # read traj
    if len(dat["traj_file"]) == 1:
        traj = mda.Universe(dat["traj_file"][0])
    else:
        traj = mda.Universe(dat["traj_file"][0], dat["traj_file"][1])

    # get cell vectors in 1d, assuming cell vectors are constant in traj
    print("Assuming cell vectors are constant in traj")
    v_cell =  traj.trajectory[0].triclinic_dimensions.reshape(-1)

    for frames, d in zip(idx, dir_list):
        n_points = len(frames)
        np.save(f"{d}/set.000/box.npy", np.tile(v_cell, (n_points,1)))

def replace_names(old_names, uni_names, sub_names):
    """replace old names with types in DP"""
    name_dict = dict(zip(uni_names, sub_names))
    res = np.array([ name_dict[name] for name in old_names ])
    return res

def get_new_types(dat):
    # read traj
    if len(dat["traj_file"]) == 1:
        traj = mda.Universe(dat["traj_file"][0])
    else:
        traj = mda.Universe(dat["traj_file"][0], dat["traj_file"][1])

    names = traj.atoms.names # get all old names
    indexes = np.unique(names, return_index=True)[1]
    uni_names = names[np.sort(indexes)] # unique without sort
    sub_names = dat["new_types"]

    new_names = replace_names(names, uni_names, sub_names)
# replace names with new types
    indexes = np.unique(new_names, return_index=True)[1]
    new_uni_names = new_names[np.sort(indexes)] # for type files
    return new_uni_names, new_names

def get_type_idx(types, names):
    N = len(types)
    idx = np.arange(N, dtype= np.int32)
    idx_dict = dict(zip(types, idx))
    res = np.array([ idx_dict[t] for t in names ], dtype= np.int32)
    return res

def mk_type(dat, dir_list):
    # make type string
    new_types, new_names = get_new_types(dat) 
    print(new_types)
    # make map
    type_idx = get_type_idx(new_types, new_names) 

    for d in dir_list:
        np.savetxt(f"{d}/type_map.raw", new_types, fmt= "%s")
        np.savetxt(f"{d}/type.raw", type_idx, fmt="%d", newline = " ")

def mk_sub_data(dat, idx, sub_prefix):
    """ make splitted data set by json data
    dat: json data
    idx: selected frames for making data 
    sub_prefix: dirname of sub data set
    """
    dir_list = mk_path(dat, idx, sub_prefix)
    idx_split = split_np(idx, dat["n_per_sub_dat"])
    # mk_coord(dat, idx_split, dir_list)
    # mk_energy(dat, idx_split, dir_list)
    # mk_box(dat, idx_split, dir_list)
    mk_type(dat, dir_list)
    return dir_list

def mk_path(dat, idx, sub_prefix):
    n_dirs = len(idx)/ dat["n_per_sub_dat"]
    idx_split = split_np(idx, dat["n_per_sub_dat"])
    n_dirs = len(idx_split)
    dir_list = []
    for i, idx_arr in enumerate(idx_split):
        prefix = dat["prefix"]+ sub_prefix+ f"{i}"
        os.makedirs(prefix, exist_ok = True)
        prefix_sub = f"{prefix}/set.000"
        os.makedirs(prefix_sub, exist_ok = True)
        dir_list.append(prefix)

    return dir_list

def mk_data_path(N, nPerSet, prefix):
# mk data path
# prefix
# | data_*
# | | set.000 type
# | | | energy, box, coord

    assert N > nPerSet, f"{N} <= {nPerSet}, cannot be divided"

# mk data_* path
    setPathList = []
    for i, x in enumerate(range(0, N, nPerSet)):
        # setPath = os.path.join(newDir, f"{prefix}data_{i:}/set.000/")
        setPath = f"{prefix}data_{i:0>3}/set.000/"
        os.makedirs(setPath, exist_ok = True)
        setPathList.append(setPath)

    boxPathList = [os.path.join(path, "box.npy") for path in setPathList]
    coordPathList = [os.path.join(path, "coord.npy") for path in setPathList]
    ePathList = [os.path.join(path, "energy.npy") for path in setPathList]

    return boxPathList, coordPathList, ePathList

def main():
    mk_dataset("./acn.json", "June20")

if __name__ == "__main__":
    main()
