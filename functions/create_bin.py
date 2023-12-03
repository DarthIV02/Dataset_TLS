# From las file create velodyne folder

import os
import laspy
from operator import itemgetter
import numpy as np
from tqdm import tqdm

def write_points_to_bin(points, bin_file):
        # Ensure the points array is of shape (-1, 4)
        if points.shape[1] != 4:
            raise ValueError("Input points array must have shape (-1, 4) for x, y, z, intensity.")

        # Convert the points array to a 1D array of float32 values
        points = points.astype(np.float32).ravel()

        # Write the points to the binary file
        points.tofile(bin_file)

def create_bin(dir_input, dir_output): # Directory of the dataset
    dirs = [f for f in os.listdir(dir_input)]
    path = [os.listdir(f'{dir_input}/{f}') for f in dirs]

    j = 1
    for i, direct in enumerate(dirs):
        for file in tqdm(path[i]):
            complete_path = os.path.join(direct, file) 
            las = laspy.read(complete_path+'/'+'Class_i_'+file+'.las')
            
            # Create velodyne file
            coords = np.vstack((las.x, las.y, las.z, las.intensity)).transpose()
            write_points_to_bin(coords, os.path.join(dir_output, f"velodyne/{str(j).zfill(6)}.bin"))
            j+=1

if __name__ == "__main__":

    create_bin("dense_dataset", "dense_dataset_semantic/sequences/00")