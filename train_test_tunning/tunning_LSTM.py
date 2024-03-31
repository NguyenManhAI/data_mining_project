import optuna
from train_test_LSTM import Trainer_Tester

def objective(trial, data, target):
  hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
  dropout = trial.suggest_float('dropout', 0, 0.5)
  num_layers = trial.suggest_int('num_layers', 1, 4)

  bidirectional = trial.suggest_categorical('bidirectional', [True, False])
  batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
  lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
  gammar = trial.suggest_loguniform('gammar', 1e-3, 1e-1)

  tiger = Trainer_Tester(data=data, target=target, model = None,hidden_dim = hidden_dim, dropout = dropout, num_layers = num_layers,
                bidirectional = bidirectional, batch_size = batch_size, lr = lr, gamma = gammar)
  _, target_value,_ = tiger.train()
  
  target_value = target_value[-1].item()
  return target_value


def Tunning(data, target, n_trials = 120):
    # tunning hyperparameter
    study = optuna.create_study()
    study.optimize(lambda trial : objective(trial=trial, data=data, target=target), n_trials=n_trials)

    # Truy cập vào giá trị tốt nhất và siêu tham số tương ứng
    best_params = study.best_params
    best_value = study.best_value

    return study.trials_dataframe(), best_params, best_value