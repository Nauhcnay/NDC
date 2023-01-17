# lift each 2d udf and labels into 3d, and save as the h5py format
# also generate the file list

import h5py
import numpy as np

import os
from os.path import join
from multiprocessing import Pool

def to_3d_h5py(path_to_udf2d, path_to_udf3d):
    if os.path.exists(path_to_udf3d): return
    print("log:\topening %s"%path_to_udf2d)
    gt_2d = np.load(path_to_udf2d)
    udf_2d = gt_2d['udf'][:257, :257]
    edge_x_2d = np.expand_dims(gt_2d['edge_x'][:257, :257], axis = -1)
    edge_y_2d = np.expand_dims(gt_2d['edge_y'][:257, :257], axis = -1)
    udf_3d = np.tile(udf_2d, (51, 1, 1)).transpose((1,2,0))
    gt_3d = np.zeros((257, 257, 51, 3))
    # we don't need the label along z direction, cause they should always be 0 in our case
    gt_3d[..., 0] = edge_x_2d
    gt_3d[..., 1] = edge_y_2d
    # write arrays into h5py
    # hdf5_file.create_dataset(str(grid_size)+"_int", [grid_size_1,grid_size_1,grid_size_1,num_of_int_params], np.uint8, compression=9)
    with h5py.File(path_to_udf3d, 'w') as f:
        f.create_dataset("256_sdf", [257, 257, 51], np.float32, compression = 9)
        f.create_dataset("256_int", [257, 257, 51, 3], np.uint8, compression = 9)
        f["256_sdf"][:] = udf_3d
        f["256_int"][:] = gt_3d

if __name__ == "__main__":
    __spec__ = None
    path_sketchvg = "./groundtruth/sketchvg"
    path_sketchvg_gt = "./groundtruth/gt_sketchvg"

    # write new file list
    flist_new = [s.strip(".npz") for s in os.listdir(path_sketchvg)]
    flist_new.sort()
    with open("abc_obj_list.txt", 'w') as f:
        f.write('\n'.join(flist_new))


    # lift each 2d udf to 3d
    gt_list = []
    for fname in flist_new:
        gt_2d = join(path_sketchvg, fname+".npz")
        gt_3d = join(path_sketchvg_gt, fname + ".hdf5")
        gt_list.append([gt_2d, gt_3d])

    with Pool(32) as pool:
        pool.starmap(to_3d_h5py, gt_list)

