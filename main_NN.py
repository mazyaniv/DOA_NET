from matplotlib import pyplot as plt
from models import *
from train_func import *
from functions_NN import *
from classes_NN import *
import torch
import numpy as np
from scipy.interpolate import interp1d
#/home/mazya/.conda/envs/yaniv/bin/python

if __name__ == "__main__":
    SNR_space = np.linspace(-5, 25, 1)#np.delete(np.linspace(-5, 25, 8), [4,6])
    N_a = [10]
    N_q = [0]
    pram = {"snap":400, "teta_range":[0,60], "D":2, "C":10}
    train_prameters = train_prameters(110, 10, 20, 3, 0.001)
    my_dict = {"device":"CPU",
               "Generate new data": True,
               "Train": False,
               "Test": False,"Plot": False}
    my_parameters = prameters_class(10, 0, pram["snap"], pram["teta_range"], pram["D"],
                                    pram["C"])  # M=N_a+N_q
# ======================================================================================================================
    if my_dict["device"] == "Cuda":
        file_path = '/home/mazya/DNN/'
    else:
        file_path = 'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'

    if my_dict["Generate new data"]:
        generate_data(my_parameters, train_prameters, file_path)

    data_file_path = file_path + 'Data/'
    my_data = My_data(data_file_path)
    print("hi")
    if my_dict["Train"]:
        my_model = LSTM(my_parameters)
        my_model.weight_init(mean=0, std=0.02)
        my_train(my_data, my_model, my_parameters, train_prameters, file_path + 'Trained_Model/', True)

    elif my_dict["Test"]:
        Error = np.zeros((len(SNR_space), len(N_a)))
        for i in range(len(SNR_space)):
            for j in range(len(N_a)):
                Model = LSTM(my_parameters)
                Model.load_state_dict(torch.load(file_path+'Trained_Model/'
                                                 +f'trained_model_N_a={my_parameters.M-my_parameters.N_q}_N_q={my_parameters.N_q}_SNR={my_parameters.SNR}.pth'))
                Model.eval()
                Error[i, j] = test_model(Model,my_data.data_test,my_data.labels_test,my_parameters.C)
        if my_dict["Plot"]:
            fig = plt.figure(figsize=(10, 6))
            colors = ['b', 'g', 'orange', 'black', 'red']
            for i in range(len(N_a)):
                if i > len(N_a) - 3:
                    style = 'dashed'
                else:
                    style = 'solid'
                cubic_interpolation_model = interp1d(SNR_space, Error[:, i], kind="slinear")
                X_ = np.linspace(SNR_space.min(), SNR_space.max(), 500)
                Y_ = cubic_interpolation_model(X_)
                plt.plot(X_, Y_, color=colors[i], linestyle=style,
                         label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
            plt.title(f"RMSE for snap={my_parameters.snap}, M={my_parameters.M}, D={my_parameters.D}")
            plt.grid()
            plt.ylabel("RMSE (Deg.)")
            plt.xlabel("SNR [dB]")
            plt.legend()
            plt.show()



