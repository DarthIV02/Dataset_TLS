# Get statistics on the data

import os
import laspy
from operator import itemgetter
import numpy as np

classes = {'unlabeled': 0, 'alive': 1, '1': 2, '10': 3, '100': 4, '1000': 5}

def Extract(lst, item):
    return list( map(itemgetter(item), lst ))

def get_mean(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    mean_x = 0
    mean_y = 0
    mean_z = 0

    full_array = []

    for i, direct in enumerate(dirs):
        for file in path[i]:
            complete_path = os.path.join(direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                full_array.append(coord)
            print(file)
            
    mean_x = np.mean(full_array, axis=0)[0]
    mean_y = np.mean(full_array, axis=0)[1]
    mean_z = np.mean(full_array, axis=0)[2]

    print(mean_x, mean_y, mean_z)

def get_std(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    std_x = 0
    std_y = 0
    std_z = 0

    full_array = []

    for i, direct in enumerate(dirs):
        for file in path[i]:
            complete_path = os.path.join(direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                full_array.append(coord)
            print(file)

    std_x = np.std(full_array, axis=0)[0]
    std_y = np.std(full_array, axis=0)[1]
    std_z = np.std(full_array, axis=0)[2]

    print(std_x, std_y, std_z)

def get_num_points_per_class(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    points_per_class = {}

    for i, direct in enumerate(dirs):
        for file in path[i]:
            complete_path = os.path.join(direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                if coord[7] not in points_per_class.keys():
                    points_per_class[coord[7]] = 1
                else:
                    points_per_class[coord[7]] += 1
            print(file)

    print(points_per_class)
        
if __name__ == "__main__":
    
    print("Mean of dataset")
    get_mean("dense_dataset")

    print("Std of dataset")
    get_std("dense_dataset")

    print("Points per class")
    get_num_points_per_class("dense_dataset")