import optuna
from Tiger_LSTM import Tiger
import pickle

def objective(trial):
  # hidden_dim = 128, dropout = 0.0, numlayers = 1, bidirectional = False, batch_size = 4, lr = 0.01, gammar = 0.1
  hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
  dropout = trial.suggest_float('dropout', 0, 0.5)
  num_layers = trial.suggest_int('num_layers', 1, 4)

  # if num_layers == 1:
  #   dropout = 0

  bidirectional = trial.suggest_categorical('bidirectional', [True, False])
  batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
  lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
  gammar = trial.suggest_loguniform('gammar', 1e-3, 1e-1)

  tiger = Tiger(model = None,hidden_dim = hidden_dim, dropout = dropout, numlayers = num_layers,
                bidirectional = bidirectional, batch_size = batch_size, lr = lr, gammar = gammar)
  _, target_value,_ = tiger.train()
  target_value = target_value[-1].item()
  return target_value


def Tunning(n_trials = 120):
    # tunning hyperparameter
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    # Truy cập vào giá trị tốt nhất và siêu tham số tương ứng
    best_params = study.best_params
    best_value = study.best_value

    return study.trials_dataframe(), best_params, best_value

    # print('Best Parameters:', best_params)
    # print('Best Value:', best_value)

    # path_params = 'best_hyperparams.pkl'

    # # Lưu trữ từ điển
    # with open(path_params, 'rb') as f:
    #     best_params = pickle.load(f)

    # print(best_params)