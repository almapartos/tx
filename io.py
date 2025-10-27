import numpy as np
import h5py

def read(filename, label="data"):
    f = h5py.File(filename, "r")
    data = np.array(f.get(label))
    return data

def write(data, filename, label="data"):
    f = h5py.File(filename, "w")  # write; can change to append if this is not the desired behavior
    f.create_dataset(label, data=data)
    f.close()

### MWE ###

def w_random_data():
    stuff = np.random.rand(8, 6)
    write(stuff, "mytestdata", label="data")

w_random_data()
data = read("mytestdata", label="data")
print(data.shape)
print(data)
