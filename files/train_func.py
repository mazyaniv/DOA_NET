import torch.optim as optim
import numpy as np
import torch
from functions import *

def my_train(data,model,parameters,train_pram,checkpoint_path,checkpoint_bool=False):
  model.train()
  optimizer = optim.Adam(model.parameters(),lr=train_pram.learning_rate, weight_decay=train_pram.weight_decay)
  for epoch in range(train_pram.epoch):
    if epoch>1:
      print(loss)
    if epoch%7 == 0:
      print(f"Epoch number {epoch}")
    #shuffle the data each epoch
    train_size = data.data_train.shape[1]
    per = np.random.permutation(train_size)
    train = data.data_train[:,per]
    labels = data.labels_train[per]

    for i in range(0, train_size, train_pram.batch):
        if (i + train_pram.batch) > train_size:
            break
        # get the input and targets of a minibatch
        z,s = get_batch(train, labels, i, i+train_pram.batch,parameters.teta_range)
        optimizer.zero_grad()
        loss = BCEWithLogitsLoss(model,z,s) # compute the total loss
        loss.backward()
        optimizer.step()

  if checkpoint_bool:
        torch.save(model.state_dict(), checkpoint_path+f'trained_model_N_a={parameters.M-parameters.N_q}_N_q={parameters.N_q}_SNR={parameters.SNR}.pth')
  print("Finish")

def get_batch(R, labels, inx_min, inx_max,teta_range):
  xt = R[:,inx_min:inx_max]
  st = labels[inx_min:inx_max]
  permutation = np.random.permutation(inx_max - inx_min)
  return xt[:,permutation] , make_onehot(st[permutation],teta_range) #Shift or Onehot

def test_model(model, data, labels,C):
    # print("Q=",Q)
    labels = labels.squeeze()
    model.eval()
    n = data.shape[1]
    z = torch.tensor(data, dtype=torch.float32).transpose(0, 1)
    with torch.no_grad():
        z = model(z)
        z = np.argsort(z.detach().numpy(), 1)[:, ::-1]
        z = z[:, :labels.shape[1]].squeeze()
        pred = np.sort(z, 1)[:, ::-1].squeeze()

        equal_elements = np.sum(np.all(pred == labels, axis=1))
        accuracy_percentage = equal_elements / n * 100.0

        sub_vec_old = pred - labels
        mask = np.logical_and(-C < np.min(sub_vec_old, axis=1), np.max(sub_vec_old, axis=1) < C)
        sub_vec_new = sub_vec_old[mask]

        RMSE = (np.sum(np.sum(np.power(sub_vec_new, 2), 1)) / (sub_vec_new.shape[0] * (pred.shape[1]))) ** 0.5
        # print(f"Accuracy: {accuracy_percentage:.2f}%")
        # print(f"RMSE : {RMSE:.2f}_Degrees,", "Number of relevant tests:",np.shape(sub_vec_new)[0])
        # print("======")
        return RMSE

def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    """Multiply a matrix Kx with its Hermitian conjecture (gram matrix),
        and adds eps to the diagonal values of the matrix,
        ensuring a Hermitian and PSD (Positive Semi-Definite) matrix.

    Args:
    -----
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N],
            where BS is the batch size and N is the matrix size.
        eps (float): Constant multiplier added to each diagonal element.
        batch_size(int): The number of batches

    Returns:
    --------
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].

    """
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