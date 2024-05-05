import numpy as np
import torch
import torch.nn as nn
import math
import random
from files.classes import *
from itertools import permutations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def observ(teta,M,SNR,snap):
    A = Matrix_class(M, teta).matrix()
    M = A.shape[0]
    D = A.shape[1]

    amplitude = 10**(SNR / 10)
    s_samp = (amplitude/np.sqrt(2))*(
                np.random.randn(D, snap)
                + 1j * np.random.randn(D, snap)
                )
    n_samp = (1/np.sqrt(2))*(
            np.random.randn(M, snap)
            + 1j * np.random.randn(M, snap))
    x_a_samp = (A @ s_samp) + n_samp
    return x_a_samp

def quantize_part(A,P,thresh=0):
        mask = np.zeros(np.shape(A),dtype=complex)
        mask[:P,:] = (1/math.sqrt(2))*(np.sign(A[:P,:].real-(thresh))+(1j*(np.sign(A[:P,:].imag-((thresh))))))
        mask[P:,:] = A[P:,:]
        return mask

def set_unified_seed(seed: int = 42): # TODO - what is it for? is it necessary?
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])

    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]), prediction.shape[0])):
        torch_perm_list.append(prediction.index_select(0, torch.tensor(list(p), dtype=torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim=0)
    return predictions
