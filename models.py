import torch.nn as nn
import torch
import numpy as np
from functions_NN import *
class CNN(nn.Module):
    def __init__(self,param,n1=12,n2=12,n3=6, kernel_size=3,padding_size=2,a=0.5):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.activation = nn.ReLU()
        self.a = a
        self.sigmo = nn.Sigmoid()
        #nn.Dropout2d
        #nn.BatchNorm2d

        #convolution layers
        self.conv1=nn.Conv2d(in_channels=2, out_channels=self.n1, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2=nn.Conv2d(in_channels=self.n1, out_channels=self.n2, kernel_size=self.kernel_size)
        self.conv3=nn.Conv2d(in_channels=self.n2, out_channels=self.n3, kernel_size=self.kernel_size)

        self.active = torch.nn.LeakyReLU(self.a)
        self.max = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        #fully-connected layers
        self.fc1 = nn.Linear(self.n3*16,1500)
        self.fc2 = nn.Linear(1500,1500)
        self.fc3 = nn.Linear(1500,param.teta_range[1]-param.teta_range[0]+1) #Resolution

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self,x):
      #1st cnn layer
      x = self.conv1(x)
      x = self.drop(x)
      x = self.active(x)
      x = self.conv2(x)
      x = self.drop(x)
      x = self.active(x)
      x = self.drop(x)
      x = self.conv3(x)
      x = self.active(x)
      x = self.max(x) #torch.Size([n3, 4, 4])
      x = x.reshape(-1,self.n3*16)
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      #x = self.sigmo(x)
      return x #torch.argmax(x,dim=1)

class LSTM(nn.Module):
    def __init__(self, param, n1=12, n3=6, kernel_size=3, padding_size=2, a=0.5):
        super(LSTM, self).__init__()
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.n1 = n1
        self.n3 = n3
        self.activation = nn.ReLU()
        self.a = a
        self.sigmo = nn.Sigmoid()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.n1, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv2d(in_channels=self.n1, out_channels=self.n3, kernel_size=self.kernel_size)

        self.active = torch.nn.LeakyReLU(self.a)
        self.max = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.5, inplace=False)

        # LSTM layer
        self.lstm_input_size = self.n3 * 4 * 4  # Adjust based on the output size after convolutions
        self.lstm_hidden_size = 64  # You can adjust this value
        self.lstm_layers = 1
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_layers, batch_first=True)

        # Fully-connected layers
        self.fc1 = nn.Linear(self.lstm_hidden_size, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, param.teta_range[1] - param.teta_range[0] + 1)  # Resolution

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x):
        # 1st convolution layer
        x = self.conv1(x)
        x = self.drop(x)
        x = self.active(x)

        # 2nd convolution layer (replaced by LSTM)
        # Reshape to fit LSTM input format
        x = x.view(x.size(0), -1, self.n3 * 4 * 4)
        # Apply LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Select the last LSTM output

        # Fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, tau: int, M: int):
        super().__init__(tau, M)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
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
        return Rz

if __name__ == "__main__":
    from classes_NN import *
    file_path = '/home/mazya/DNN/Data/' #'C:/Users/Yaniv/PycharmProjects/DOA/DNN/Data/'
    data_train = np.load(file_path + 'data_test_N_a=5_N_q=5_SNR=25.0.npy')
    x = data_train
    x = torch.tensor(x, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    x = x[0:4]
    my_parameters = prameters_class(10, 5, 2, 400, [0,60], 2, 10)
    z = LSTM(my_parameters)(x)
    print(z.shape) #torch.Size([4, 61])
