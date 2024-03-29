# Tunning, save best hyperparameter, save best parameter for model

from Tunning import Tunning
import pickle
from Tiger_LSTM import Tiger
from PrepareData import *
import torch

def execute():
    # Tuuning
    data_frame,best_hyperparams, best_value = Tunning()

    # lưu kết quả
    path_result = 'Data\\result_tunning.csv'
    data_frame.to_csv(path_result, header = True, index = False, mode = 'w')

    #lưu best parameter

    path_hyperparams = 'Data\\best_hyperparams.pkl'

    # Lưu trữ từ điển
    with open(path_hyperparams, 'wb') as f:
        pickle.dump(best_hyperparams,f)

    tiger = Tiger(None,best_hyperparams['hidden_dim'],best_hyperparams['dropout'],best_hyperparams['num_layers'],best_hyperparams['bidirectional'],
                best_hyperparams['batch_size'],best_hyperparams['learning_rate'],best_hyperparams['gammar'])
    train_loss ,val_loss,_ = tiger.train(earlystop_train = True)

    # save best parameter for model
    path_best_parameter_model = 'Data\\best_params_model.pkl'
    torch.save(tiger.get_model().state_dict(), path_best_parameter_model)
    print("Done")


if __name__ == "__main__":
    execute()