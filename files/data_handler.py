import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from files.classes import *
from files.functions import *
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def generate_data(array: array_class,save_datasets: bool = False,NN: bool = False,file_path: Path = Path.cwd(),phase: str="train_samples"):
    generic_dataset = []
    model_dataset = []
    for i in tqdm(range(array.J)):
        if array.D == 1:
            teta = np.random.randint(array.teta_range[0], array.teta_range[1], size=array.D)
        else:
            while True:
                teta = np.random.randint(array.teta_range[0], array.teta_range[1], size=array.D)
                if teta[0] != teta[1]:
                    break
            teta = np.radians(np.sort(teta)[::-1])
        sample = quantize_part(observ(teta, array.M, array.SNR, array.snap),array.N_q) #quantize

        X =torch.tensor(sample, dtype=torch.complex64)
        Y = torch.tensor(teta.copy(), dtype=torch.float64)
        generic_dataset.append((X, Y))
        if NN:
            X_model = create_autocorrelation_tensor(X, 8).to(torch.float)
            model_dataset.append((X_model, Y))

    if save_datasets:
        torch.save(obj=model_dataset, f=file_path / phase / f'data_{array.J}_samples_train.npy')
        torch.save(obj=generic_dataset, f=file_path / phase / f'data_{array.J}_samples_train_generic.npy')
    return model_dataset, generic_dataset
# ======================================================================================================================

def autocorrelation_matrix(X: torch.Tensor, lag: int):
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr

def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    # Insuring Tensor input
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)

    Kx_list = []
    bs_kx = Kx
    for iter in range(batch_size):
        K = bs_kx[iter]
        # Hermitian conjecture
        Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)
        # Diagonal loading
        eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
        Rz = Kx_garm + eps_addition
        Kx_list.append(Rz)
    Kx_Out = torch.stack(Kx_list, dim=0)
    return Kx_Out