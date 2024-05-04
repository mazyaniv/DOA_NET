import torch
import torch.nn as nn
from files.data_handler import *
from files.criterions import *
def evaluate_dnn_model(model,dataset: list):
    overall_loss = 0.0
    test_length = 0
    model.eval()
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            model_output = model(X)
            DOA_predictions = model_output[0]
            eval_loss = Loss(DOA_predictions, DOA)
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    return overall_loss