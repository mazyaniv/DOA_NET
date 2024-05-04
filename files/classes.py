import numpy as np
import torch

class array_class():
    def __init__(self, M,N_q,J,snapshot,teta_range,D,SNR=10):
        self.M = M
        self.N_q = N_q
        self.J = J #DATA SIZE
        self.snap = snapshot
        self.teta_range = teta_range
        self.D = D
        self.SNR = SNR
class train_prameters():
    def __init__(self, test_size,batch,epoch,learning_rate, weight_decay=1e-9):
        self.test_size = test_size
        self.batch = batch
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
class Matrix_class():
    def __init__(self, M, teta):
        self.teta = teta
        self.M = M
        self.D = len(teta)
        self.A = np.zeros((self.M, self.D), dtype=complex)
    def matrix(self):
        # teta = np.radians(self.teta)
        A_mask = np.zeros((self.M, self.D), dtype=complex)
        for j in range(self.D):
            A_mask[:, j] = np.exp(-1j * np.pi * np.arange(self.M) * np.sin(self.teta[j]))
        self.A = A_mask
        return self.A

class My_data():
    def __init__(self, file_path):
        self.data_train = torch.load(file_path/'data_train.npy')
        # self.data_test = np.load(file_path/'data_test.npy')
        # self.labels_test = np.load(file_path/'labels_test.npy')

if __name__ == "__main__":
    print("Not main file")