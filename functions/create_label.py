# From las file create labels folder

import os
import laspy
import numpy as np
from tqdm import tqdm

def create_label(dir_input, dir_output): # Directory of the dataset
    dirs = [f for f in os.listdir(dir_input)]
    dirs.sort()
    path = [os.listdir(f'{dir_input}/{f}') for f in dirs]

    # Write labels

    for i, direct in enumerate(dirs):
        j = 1
        for file in tqdm(path[i]):
            complete_path = os.path.join(dir_input, direct, file)
            las = laspy.read(complete_path+'/'+'Class_i_'+file+'.las') # Read file
            labels = np.vstack((las.classification)).transpose().astype('uint32') # Stack classification
            output = os.path.join(dir_output,f"0{i}/labels/{str(j).zfill(6)}.label")
            np.save(output, labels[0])
            j+=1

if __name__ == "__main__":

    create_label("dense_dataset", "dense_dataset_semantic/sequences")
