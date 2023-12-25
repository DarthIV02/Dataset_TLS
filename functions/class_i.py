# Combine the classified scan with intensity values

import os
import laspy
from operator import itemgetter
import numpy as np
from tqdm import tqdm

def Extract(lst, item):
    return list( map(itemgetter(item), lst ))

def combine_class_i(dir): # Directory of the dataset
    dirs = [f for f in os.listdir(dir)]
    path = [os.listdir(f'{dir}/{f}') for f in dirs]

    for i, direct in enumerate(dirs):
        for file in path[i]:
            complete_path = os.path.join(dir, direct, file) 
            if(not os.path.exists(complete_path+'/'+f'Class_i_{file}.las')):
                las_complete = laspy.read(complete_path + '/'+file+'.las')
                print(complete_path + '/'+file+'.las')
                coords = np.vstack((las_complete.x, las_complete.y, las_complete.z, las_complete.red, las_complete.green, las_complete.blue, las_complete.intensity)).transpose()
                #coords = coords.astype('float16')

                las = laspy.read(complete_path+'/Classified_scan_'+file+'.las')
                coords_clas = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue, las.classification)).transpose()
                #coords_clas = coords_clas.astype('float16')

                output_file_name = f'Class_i_{file}.las'
                header = laspy.LasHeader(point_format=3, version="1.2")
                outfile = laspy.LasData(header)

                data = []
                for x, y, z, r, g, b, label in tqdm(coords_clas):
                    matching_indices = np.where((coords[:, 0] == float(x)) & (coords[:, 1] == float(y)) & (coords[:, 2] == float(z)))
                    try:
                        matching_indices = matching_indices[0]
                        #closest = np.argmin(np.abs(coords[matching_indices][:][2] - z))
                        #intensity = coords[matching_indices[closest]][3]
                        intensity = coords[matching_indices][3]
                        point = (x, y, z, int(r), int(g), int(b), intensity, label)
                        data.append(point)
                    except:
                        #print(f"Didn't pass: {x}, {y}, {z}")
                        pass

                # Add points to the LAS file
                outfile.x, outfile.y, outfile.z = [Extract(data, i) for i in range(3)]
                outfile.red, outfile.green, outfile.blue = [Extract(data, i) for i in range(3, 6)]
                outfile.intensity = Extract(data, 6)
                outfile.classification = Extract(data, 7)
                #outfile.Classification = [p[6] for p in data]

                # Write to LAS file
                outfile.write(os.path.join(complete_path, output_file_name))
                print(os.path.join(complete_path, output_file_name))
            else:
                print("Skipped: ", file)

if __name__ == "__main__":

    combine_class_i("dense_dataset")
