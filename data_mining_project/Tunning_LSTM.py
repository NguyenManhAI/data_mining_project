import optuna
from Train_Test_LSTM import TrainTest
from Preparedata_LSTM import Data

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

  train_test = TrainTest(hidden_dim = hidden_dim, dropout = dropout, numlayers = num_layers,
                bidirectional = bidirectional, batch_size = batch_size, lr = lr, gammar = gammar)
  _, target_value,_ = train_test.train()
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

if __name__ == "__main__":
    tunning = Tunning(10)
    print(tunning)