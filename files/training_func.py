from pathlib import Path
import torch
from files.functions import set_unified_seed
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import copy
from files.criterions import Loss
from files.evaluation import evaluate_dnn_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(training_parameters,train_data,model,plot_curves: bool = True,saving_path: Path = None,model_name=None):
    # Set the seed for all available random operations
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    model, loss_train_list, loss_valid_list = train_model(training_parameters,train_data,model)
    # Save models best weights
    file_name = f"{len(train_data)} samples "+ str(Path(dt_string_for_save))
    torch.save(model.state_dict(), saving_path/file_name)
    # Plot learning and validation loss curves
    if plot_curves:
        plot_learning_curve(list(range(training_parameters.epoch)), loss_train_list, loss_valid_list)
    return model, loss_train_list, loss_valid_list

def train_model(training_params,train_dataset,model,model_name=None,checkpoint_path=None):
    print(type(model))
    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.1, shuffle=True)
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=training_params.batch, shuffle=True,drop_last=False)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    schedular = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2) #TODO
    # Initialize losses
    loss_train_list = []
    loss_valid_list = []
    min_valid_loss = np.inf
    # Set initial time for start training
    since = time.time()
    print("\n---Start Training Stage ---\n")
    # Run over all epochs
    for epoch in range(training_params.epoch):
        train_length = 0
        overall_train_loss = 0.0
        # Set model to train mode
        model.train()
        model = model.to(device)
        for data in tqdm(train_dataset):
            Rx, DOA = data
            train_length += DOA.shape[0]
            # Cast observations and DoA to Variables
            Rx = Variable(Rx, requires_grad=True).to(device) #TODO- is it necessary?
            DOA = Variable(DOA, requires_grad=True).to(device)
            # Get model output
            model_output = model(Rx)
            DOA_predictions = model_output[0]
            train_loss = Loss(DOA_predictions, DOA)
            # Back-propagation stage
            try:
                train_loss.backward()
            except RuntimeError:
                print("linalg error")
            # optimizer update
            optimizer.step()
            # reset gradients
            model.zero_grad()
            # add batch loss to overall epoch loss
            overall_train_loss += train_loss.item() # RMSPE is summed
        # Average the epoch training loss
        overall_train_loss = overall_train_loss / train_length
        loss_train_list.append(overall_train_loss)
        # Update schedular
        schedular.step()
         # Calculate evaluation loss
        valid_loss = evaluate_dnn_model(valid_dataset,model)
        loss_valid_list.append(valid_loss)
        # Report results
        print("epoch : {}/{}, Train loss = {:.6f}, Validation loss = {:.6f}".format(epoch + 1, training_params.epoch, overall_train_loss, valid_loss))
        print("lr {}".format(optimizer.param_groups[0]["lr"]))
        # Save best model weights for early stoppings
        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model")
            min_valid_loss = valid_loss
            best_epoch = epoch
            # Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model.state_dict(), checkpoint_path / model_name)

    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Minimal Validation loss: {:4f} at epoch {}".format(min_valid_loss, best_epoch))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), checkpoint_path / model_name)
    return model, loss_train_list, loss_valid_list

def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list):
    plt.title("Learning Curve: Loss per Epoch")
    plt.plot(epoch_list, train_loss, label="Train")
    plt.plot(epoch_list, validation_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()



