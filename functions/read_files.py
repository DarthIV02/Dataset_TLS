import numpy as np
from operator import itemgetter
import laspy

def Extract(lst, item):
    return list( map(itemgetter(item), lst ))

def read_points(bin_file):
    points = np.fromfile(bin_file, dtype = np.float32)
    points = np.reshape(points,(-1,4)) # x,y,z,intensity
    points = points[8:]
    return points

def read_labels(label_file):
    label = np.load(label_file)
    print("LABEL", label)
    #label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    return sem_label.astype(np.int32)

def semantic_pointcloud(points, file):

    # Create las file
    output_file_name = f'{file}.las'
    header = laspy.LasHeader(point_format=3, version="1.2")
    outfile = laspy.LasData(header)

    search = []
    for x,y,z,intensity in points:
        point = (x, y, z, intensity)
        search.append(point)
    outfile.x, outfile.y, outfile.z, outfile.intensity = [Extract(search, i) for i in range(4)]
    outfile.write(output_file_name)

if __name__ == "__main__":
    #semantic_pointcloud(read_points("000487.bin"), "000487")
    read_points('Dataset_TLS/dense_dataset_numpy/sequences/00/velodyne/000001.bin.npy')
    read_labels('Dataset_TLS/dense_dataset_numpy/sequences/00/labels/000001.label.npy')