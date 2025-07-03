import numpy as np

def compare_npy(received_tensor, residing_tensor_path):
    t2 = np.load(residing_tensor_path)
    t1 = received_tensor.flatten()
    t2 = t2.flatten()
    for i,j in zip(t1, t2):
        print(i,j)
