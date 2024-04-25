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
        if not os.path.exists(bin_file[:-10]):
            # Create the directory
            os.makedirs(bin_file[:-10])
        # np.save(bin_file, points)
        points.tofile(bin_file)

def create_bin(dir_input, dir_output): # Directory of the dataset
    dirs = [f for f in os.listdir(dir_input)]
    dirs.sort()
    print(dirs)
    path = [os.listdir(f'{dir_input}/{f}') for f in dirs]

    for i, direct in enumerate(dirs):
        j = 1
        for file in tqdm(path[i]):
            complete_path = os.path.join(dir_input, direct, file)
            las = laspy.read(complete_path+'/'+'Sub_2_'+file+'.las')
                
            # Create velodyne file
            coords = np.vstack((las.x, las.y, las.z, las.intensity)).transpose()
            coords = coords[(np.sqrt((coords[:,0])**2 + (coords[:,1])**2 + (coords[:,2])**2)) < 10]
            write_points_to_bin(coords, os.path.join(dir_output, f"0{i}/velodyne/{str(j).zfill(6)}.bin"))
            j+=1

if __name__ == "__main__":

    create_bin("Dataset_TLS/dense_dataset", "Dataset_TLS/dense_dataset_subsample_2/sequences/")
