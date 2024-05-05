from files.functions import *
from files.classes import *
from files.data_handler import *
import torch
from pathlib import Path
from files.data_handler import *
from files.training_func import *
from files.models import *
from datetime import datetime
from files.evaluation import *
R2D = 180 / np.pi
#/home/mazya/.conda/envs/yaniv/bin/python

if __name__ == "__main__":
    pram = {"snap":300, "teta_range":[-60,60], "D":2}
    train_array = array_class(10, 0, 5, pram["snap"], pram["teta_range"], pram["D"])
    my_dict = {"Generate new data": False,
               "Train": True,
               "Test": False,"Plot": False}
    train_prameters = train_prameters(20, 1, 1, 0.00001, 1e-9)
# ======================================================================================================================
    file_path = Path.cwd()/"data"
    file_path.mkdir(parents=True, exist_ok=True)
    train_path = file_path / 'train_samples'
    train_path.mkdir(parents=True, exist_ok=True)

    model_path = file_path / 'trained_model'
    model_path.mkdir(parents=True, exist_ok=True)

    if my_dict["Generate new data"]:
            hard_model,easy_model = generate_data(train_array, True,True,file_path= file_path,phase= 'train_samples')
    train_dataset = My_data(train_path,train_array).data_train
    print(len(train_dataset))
    if my_dict["Train"]:
        model = SubspaceNet(8,train_array.D)
        print(type(model))
        train(train_prameters,train_dataset,model,saving_path=model_path)
        # my_model = LSTM(my_parameters)
        # my_model.weight_init(mean=0, std=0.02)
        # my_train(my_data, my_model, my_parameters, train_prameters, file_path + 'Trained_Model/', True)

    elif my_dict["Test"]:
        chosen_date = datetime(2024, 5, 4,13,16)
        date_string = chosen_date.strftime("%d_%m_%Y_%H_%M")
        Model = SubspaceNet(8, train_array.D)
        Model.load_state_dict(torch.load(model_path / date_string))
        Model.eval()

        SNR_space = np.linspace(-5, 10, 12)
        N_a = [10]
        N_q = [10]
        Error = np.zeros((len(SNR_space), len(N_a)))
        Error1 = np.zeros((len(SNR_space), len(N_a)))
        for i in range(len(SNR_space)):
            for j in range(len(N_a)):
                test_pram = array_class(N_a[j], N_q[j], 100, 400, pram["teta_range"], pram["D"],SNR_space[i])
                datamodel_test, generic_test = generate_data(test_pram, NN=False)
                # Error[i, j] = evaluate_dnn_model(datamodel_test,Model)
                Error1[i, j] = evaluate_model_based(generic_test,test_pram)

        if my_dict["Plot"]:
            fig = plt.figure(figsize=(10, 6))
            colors = ['b', 'g', 'orange', 'black', 'red']
            # plt.plot(SNR_space, R2D*Error, color='blue', linestyle='solid', marker='o', label='SubspaceNet')
            plt.plot(SNR_space, R2D*Error1, color='red', linestyle='solid', marker='o', label='ESPRIT')
            plt.title(f"RMSE for snap={test_pram.snap}, M={test_pram.M}, D={test_pram.D}")
            plt.grid()
            plt.ylabel("RMSE (Deg.)")
            plt.xlabel("SNR [dB]")
            plt.legend()
            plt.show()



