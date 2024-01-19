# Get statistics on the data

import os
import laspy
from operator import itemgetter
import numpy as np
from tqdm import tqdm

classes = {'unlabeled': 0, 'alive': 1, '1': 2, '10': 3, '100': 4, '1000': 5}

def Extract(lst, item):
    return list( map(itemgetter(item), lst ))

def get_mean(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    dirs.sort()
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    mean_x = 0
    mean_y = 0
    mean_z = 0

    full_array = []

    for i, direct in enumerate(dirs):
        for file in tqdm(path[i]):
            complete_path = os.path.join(dir, direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                full_array.append(coord)
            
    mean_x = np.mean(full_array, axis=0)[0]
    mean_y = np.mean(full_array, axis=0)[1]
    mean_z = np.mean(full_array, axis=0)[2]

    print(mean_x, mean_y, mean_z)
    return mean_x, mean_y, mean_z

def get_std(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    std_x = 0
    std_y = 0
    std_z = 0

    full_array = []

    for i, direct in enumerate(dirs):
        for file in tqdm(path[i]):
            complete_path = os.path.join(dir, direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                full_array.append(coord)

    std_x = np.std(full_array, axis=0)[0]
    std_y = np.std(full_array, axis=0)[1]
    std_z = np.std(full_array, axis=0)[2]

    print(std_x, std_y, std_z)
    return std_x, std_y, std_z

def get_num_points_per_class(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    points_per_class = {}

    for i, direct in enumerate(dirs):
        for file in tqdm(path[i]):
            complete_path = os.path.join(dir, direct, file) 
                
            # Read labeled scan
            las = laspy.read(complete_path+'/Class_i_'+file+'.las')
            coords = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity, las.classification)).transpose().tolist()
            for coord in coords:
                if coord[7] not in points_per_class.keys():
                    points_per_class[coord[7]] = 1
                else:
                    points_per_class[coord[7]] += 1

    return points_per_class

def get_ratio_per_class(dir): # GEt percentage of each class of points
    
    total_dic = get_num_points_per_class(dir)
    total = 0
    for val in total_dic.values():
        total += val
    
    print("total =", total) 
    ratio = {}
    
    for key in total_dic.keys():
        ratio[key] = total_dic[key]/total

    return ratio
        
if __name__ == "__main__":
    
    #print("Mean of dataset")
    #print(get_mean("dense_dataset"))

    #print("Std of dataset")
    #get_std("dense_dataset")

    print("Points per class")
    print(get_num_points_per_class("Dataset_TLS/dense_dataset"))

    #print("Ratio per class")
    #print(get_ratio_per_class("dense_dataset"))
