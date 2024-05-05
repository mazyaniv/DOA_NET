import torch
import torch.nn as nn
from files.data_handler import *
from files.criterions import *
from files.models import *
R2D = 180 / np.pi
def evaluate_dnn_model(dataset: list,model):
    overall_loss = 0.0
    test_length = 0
    model.eval()
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # X = torch.unsqueeze(X, dim=0).to(device)
            DOA = torch.unsqueeze(DOA, dim=0).to(device)
            # Get model output
            model_output = model(X)
            DOA_predictions = model_output[0]
            eval_loss = Loss(DOA_predictions, DOA)
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    return overall_loss

def evaluate_model_based(dataset: list,system_model):
    loss_list = []
    for i, data in enumerate(dataset):
        X, doa = data
        doa = torch.unsqueeze(doa, dim=0).to(device)
        # X = X[0]
        R = torch.tensor(np.cov(X))
        R = torch.unsqueeze(R, dim=0).to(device)
        predictions = esprit(R, system_model.D, 1)
        loss = Loss(predictions, doa)
        loss_list.append(loss)
    return np.mean(loss_list)
