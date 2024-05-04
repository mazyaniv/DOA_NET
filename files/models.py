import torch.nn as nn
import torch
import numpy as np
from files.functions import *
from files.data_handler import *

class SubspaceNet(nn.Module):
    def __init__(self, tau: int, D: int):
        super(SubspaceNet, self).__init__()
        self.D = D
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        # Set the subspace method for training
        self.diff_method = esprit #TODO

    def anti_rectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=self.batch_size)  # Shape: [Batch size, N, N]
        doa_prediction = esprit(Rz, self.D, self.batch_size)
        return doa_prediction, Rz

def esprit(Rz: torch.Tensor, D: int, batch_size: int):
    doa_batches = []

    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        # Get signal subspace
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :D]
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (Us[0 : R.shape[0] - 1],Us[1 : R.shape[0]])
        # Generate Phi matrix
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        # Calculate the phase component of the roots
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        # Calculate the DoA out of the phase component
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)
    # eigvals, eigvecs = np.linalg.eig(R)
    #     sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    #     eigvecs_sorted = eigvecs[:, sorted_indices]
    #     Es = eigvecs_sorted[:, :pram.D]
    #     S1 = Es[1:,:]
    #     S2 = Es[:-1,:]
    #     P = np.linalg.inv(S1.conj().transpose()@S1)@S1.conj().transpose()@S2 #LS
    #     eigvals, eigvecs = np.linalg.eig(P)
    #     pred = np.degrees(np.arcsin(np.angle(eigvals) / math.pi))
    #     pred = np.sort(pred)[::-1]
    #     return pred
# class CNN(nn.Module):
#     def __init__(self,param,n1=12,n2=12,n3=6, kernel_size=3,padding_size=2,a=0.5):
#         super(CNN, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding_size = padding_size
#         self.n1 = n1
#         self.n2 = n2
#         self.n3 = n3
#         self.activation = nn.ReLU()
#         self.a = a
#         self.sigmo = nn.Sigmoid()
#         #nn.Dropout2d
#         #nn.BatchNorm2d
#
#         #convolution layers
#         self.conv1=nn.Conv2d(in_channels=2, out_channels=self.n1, kernel_size=self.kernel_size, padding=self.padding_size)
#         self.conv2=nn.Conv2d(in_channels=self.n1, out_channels=self.n2, kernel_size=self.kernel_size)
#         self.conv3=nn.Conv2d(in_channels=self.n2, out_channels=self.n3, kernel_size=self.kernel_size)
#
#         self.active = torch.nn.LeakyReLU(self.a)
#         self.max = nn.MaxPool2d(2)
#         self.drop = nn.Dropout(p=0.5, inplace=False)
#         #fully-connected layers
#         self.fc1 = nn.Linear(self.n3*16,1500)
#         self.fc2 = nn.Linear(1500,1500)
#         self.fc3 = nn.Linear(1500,param.teta_range[1]-param.teta_range[0]+1) #Resolution
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(mean, std)
#                 m.bias.data.zero_()
#
#     def forward(self,x):
#       #1st cnn layer
#       x = self.conv1(x)
#       x = self.drop(x)
#       x = self.active(x)
#       x = self.conv2(x)
#       x = self.drop(x)
#       x = self.active(x)
#       x = self.drop(x)
#       x = self.conv3(x)
#       x = self.active(x)
#       x = self.max(x) #torch.Size([n3, 4, 4])
#       x = x.reshape(-1,self.n3*16)
#       x = self.fc1(x)
#       x = self.fc2(x)
#       x = self.fc3(x)
#       #x = self.sigmo(x)
#       return x #torch.argmax(x,dim=1)
#
# class LSTM(nn.Module):
#     def __init__(self, param, n1=12, n3=6, kernel_size=3, padding_size=2, a=0.5):
#         super(LSTM, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding_size = padding_size
#         self.n1 = n1
#         self.n3 = n3
#         self.activation = nn.ReLU()
#         self.a = a
#         self.sigmo = nn.Sigmoid()
#
#         # Convolution layers
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.n1, kernel_size=self.kernel_size,
#                                padding=self.padding_size)
#         self.conv3 = nn.Conv2d(in_channels=self.n1, out_channels=self.n3, kernel_size=self.kernel_size)
#
#         self.active = torch.nn.LeakyReLU(self.a)
#         self.max = nn.MaxPool2d(2)
#         self.drop = nn.Dropout(p=0.5, inplace=False)
#
#         # LSTM layer
#         self.lstm_input_size = self.n3 * 4 * 4  # Adjust based on the output size after convolutions
#         self.lstm_hidden_size = 64  # You can adjust this value
#         self.lstm_layers = 1
#         self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_layers, batch_first=True)
#
#         # Fully-connected layers
#         self.fc1 = nn.Linear(self.lstm_hidden_size, 1500)
#         self.fc2 = nn.Linear(1500, 1500)
#         self.fc3 = nn.Linear(1500, param.teta_range[1] - param.teta_range[0] + 1)  # Resolution
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(mean, std)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         # 1st convolution layer
#         x = self.conv1(x)
#         x = self.drop(x)
#         x = self.active(x)
#
#         # 2nd convolution layer (replaced by LSTM)
#         # Reshape to fit LSTM input format
#         x = x.view(x.size(0), -1, self.n3 * 4 * 4)
#         # Apply LSTM
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Select the last LSTM output
#
#         # Fully-connected layers
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

if __name__ == "__main__":
    from classes import *
    file_path = '/home/mazya/DNN/Data/' #'C:/Users/Yaniv/PycharmProjects/DOA/DNN/Data/'
    data_train = np.load(file_path + 'data_test_N_a=5_N_q=5_SNR=25.0.npy')
    x = data_train
    x = torch.tensor(x, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    x = x[0:4]
    my_parameters = prameters_class(10, 5, 2, 400, [0,60], 2, 10)
    z = LSTM(my_parameters)(x)
    print(z.shape) #torch.Size([4, 61])
