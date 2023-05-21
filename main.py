from itertools import product
import progressbar
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

def create_bar(max_value):
  widgets = [' [',
           progressbar.Counter(format='%(value)d/%(max_value)d'), '',
           '] (',
           progressbar.ETA(), ') ',
          ]
 
  bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets).start()
  return bar

def print_params(accuracy, param_grid: dict):
  print("\nBest Params:")
  print(f"\tAccuracy: {accuracy * 100}%")
  for key in list(param_grid.keys()):
    print(f"\t{key}: {param_grid[key]}")

def find_best_params(param_grid: dict, model, X, y, preprocess):
  if preprocess:
    model_X_train = preprocessing.scale(X)
  else:
    model_X_train = X

  original_times_left = len(list(product(*param_grid.values())))
  times_left = original_times_left
  bar = create_bar(original_times_left)

  best_score = 0
  best_params = {}

  scores_num = 5

  for params in product(*param_grid.values()):
      params_dict = dict(zip(param_grid.keys(), params))
      model = type(model)(**params_dict)
      scores = cross_val_score(model, model_X_train[::10], y[::10], cv=scores_num)
      accuracy = sum(scores)/scores_num

      times_left -= 1
      bar.update(original_times_left - times_left)

      if accuracy > best_score:
          best_score = accuracy
          best_params = params_dict

  print_params(best_score, best_params)
