import torch
import torch.nn as nn
import numpy as np
from files.data_handler import *
from files.functions import *
def compute_rmse(model,tensor1, tensor2): ####TODO####

    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    z = z.detach().numpy()
    z = np.argsort(z,1)[:,::-1]
    z = np.radians(z[:,:D].squeeze()) #sort if D>1
    z = torch.tensor(z.copy().squeeze(), requires_grad=True,dtype=torch.float32)
    s = torch.Tensor(np.radians(tensor2)).squeeze()
    mse = torch.mean((z - s)**2)
    return torch.sqrt(mse)

def MSE_loss(model,tensor1, tensor2):
    z = torch.tensor(tensor1, requires_grad=True,dtype=torch.float32).transpose(0, 1)
    z = model(z)
    s = torch.tensor(tensor2.squeeze(), requires_grad=True,dtype=torch.float32)
    return nn.MSELoss()(z,s)
# z,s = get_batch(data_train_vec[0], labels_train_vec[0], 0, 2)
# print(BCEWithLogitsLoss(CNN(12,12,6),z,s))
def Loss(doa_predictions,doa):
    rmspe = []
    for iter in range(doa_predictions.shape[0]):
        rmspe_list = []
        batch_predictions = doa_predictions[iter].to(device)
        targets = doa[iter].to(device)
        prediction_perm = permute_prediction(batch_predictions).to(device)
        for prediction in prediction_perm:
            # Calculate error with modulo pi
            error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
            # Calculate RMSE over all permutations
            rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
            rmspe_list.append(rmspe_val)
        rmspe_tensor = torch.stack(rmspe_list, dim = 0)
        # Choose minimal error from all permutations
        rmspe_min = torch.min(rmspe_tensor)
        rmspe.append(rmspe_min)
    result = torch.sum(torch.stack(rmspe, dim = 0))
    return result