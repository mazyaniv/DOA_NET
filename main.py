from files.functions import *
from files.classes import *
from files.data_handler import *
import torch
from pathlib import Path
from files.data_handler import *
from files.training_func_new import *
from files.models import *
from datetime import datetime
#/home/mazya/.conda/envs/yaniv/bin/python

if __name__ == "__main__":
    pram = {"snap":200, "teta_range":[-60,60], "D":3}
    my_parameters = prameters_class(8, 0, 100, pram["snap"], pram["teta_range"], pram["D"])

    my_dict = {"Generate new data": True,
               "Train": True,
               "Test": False,"Plot": False}
    train_prameters = train_prameters(20, 10, 50, 0.00001, 1e-9)
# ======================================================================================================================
#     if my_dict["device"] == "Cuda":
#         file_path = '/home/mazya/DNN/'
#     else:
#         file_path = 'C:/Users/Yaniv/PycharmProjects/DOA/DNN/'
    file_path = Path.cwd()/"data"
    file_path.mkdir(parents=True, exist_ok=True)
    train_path = file_path / 'train'
    train_path.mkdir(parents=True, exist_ok=True)
    model_path = file_path / 'trained_model'
    model_path.mkdir(parents=True, exist_ok=True)

    if my_dict["Generate new data"]:
            hard_model,east_model = generate_data(my_parameters,True, file_path,"train")
    train_dataset = My_data(train_path).data_train[:100]

    if my_dict["Train"]:
        model = SubspaceNet(8,my_parameters.D)
        train(train_prameters,train_dataset,model,saving_path=model_path)
        # my_model = LSTM(my_parameters)
        # my_model.weight_init(mean=0, std=0.02)
        # my_train(my_data, my_model, my_parameters, train_prameters, file_path + 'Trained_Model/', True)

    elif my_dict["Test"]:
        chosen_date = datetime(2024, 5, 4,10,19)
        date_string = chosen_date.strftime("%d_%m_%Y_%H_%M")
        Model = SubspaceNet(8, my_parameters.D)
        Model.load_state_dict(torch.load(model_path / date_string))
        Model.eval()

        # Error = np.zeros((len(SNR_space), len(N_a)))
        # for i in range(len(SNR_space)):
        #     for j in range(len(N_a)):

        #         Error[i, j] = test_model(Model,my_data.data_test,my_data.labels_test,my_parameters.C)
        #
        #
        #
        # if my_dict["Plot"]:
        #     fig = plt.figure(figsize=(10, 6))
        #     colors = ['b', 'g', 'orange', 'black', 'red']
        #     for i in range(len(N_a)):
        #         if i > len(N_a) - 3:
        #             style = 'dashed'
        #         else:
        #             style = 'solid'
        #         cubic_interpolation_model = interp1d(SNR_space, Error[:, i], kind="slinear")
        #         X_ = np.linspace(SNR_space.min(), SNR_space.max(), 500)
        #         Y_ = cubic_interpolation_model(X_)
        #         plt.plot(X_, Y_, color=colors[i], linestyle=style,
        #                  label=f'Analog={N_a[i]}, Quantize={N_q[i]}')
        #     plt.title(f"RMSE for snap={my_parameters.snap}, M={my_parameters.M}, D={my_parameters.D}")
        #     plt.grid()
        #     plt.ylabel("RMSE (Deg.)")
        #     plt.xlabel("SNR [dB]")
        #     plt.legend()
        #     plt.show()
        #


