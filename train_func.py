import torch.optim as optim
import numpy as np
import torch
from functions_NN import *

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