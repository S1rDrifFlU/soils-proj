import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statistics as stat
import statistics as st
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import pearsonr
from skopt import gp_minimize, gbrt_minimize
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR

def preprocess_data(df, columns_to_drop):
  """
  Предварительная обработка данных.

  Убирает указанные столбцы и преобразует столбец "Высота" в числовой формат.

  Args:
      df (pandas.DataFrame): Исходный датасет.
      columns_to_drop (list): Список столбцов для удаления.

  Returns:
      pandas.DataFrame: Очищенный датасет.
  """
  df = df.drop(columns=columns_to_drop)

  # Это излишнее действие, так как я бы хотел избегать вариантов, когда подается не подготовленный или неисправленный набор данных
  df["Altitude"] = pd.to_numeric(df["Altitude"].str.split(",", expand=True)[0], errors="coerce") 
  return df

"""
  Пример использования
      columns_to_drop = ['irrelevant_column1', 'irrelevant_column2']
      df = preprocess_data(df.copy(), columns_to_drop)
  """


def perform_correlation_analysis(df):
  """
  Анализ корреляции признаков.

  Строит тепловую карту корреляции для признаков в датасете.

  Args:
      df (pandas.DataFrame): Исходный датасет.
  """
  correlation_matrix = df.corr()
  plt.figure(figsize=(12, 10))
  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
  plt.title("Correlation Matrix")
  plt.show()
  """
  Пример использования
      perform_correlation_analysis(df)
  """


def split_data(X, y, test_size=0.2, random_state=42):
  """
  Разделение данных на обучающую и тестовую выборки.

  Args:
      X (pandas.DataFrame): О features (независимые переменные).
      y (pandas.Series): Target variable (зависимая переменная).
      test_size (float, optional): Доля тестовой выборки. Defaults to 0.2.
      random_state (int, optional): Рандомный поток для обеспечения воспроизводимости. Defaults to 42.

  Returns:
      tuple: Кортеж из четырех элементов - X_train, X_test, y_train, y_test.
  """
  return train_test_split(X, y, test_size=test_size, random_state=random_state)



def define_regressors():
  """
  Определение словаря с используемыми алгоритмами регрессии.

  Returns:
      dict: Словарь, где ключи - названия алгоритмов, а значения - классы алгоритмов.
  """
  regressors = {
    'rf': RandomForestRegressor,
    'lgbm': LGBMRegressor,
    'xgb': XGBRegressor,
    'svm': SVR,
    'ridge': Ridge,
    'lasso': Lasso,
    'elastic_net': ElasticNet,
  }
  return regressors


def define_search_spaces(regressors):
    """
  Определение диапазонов поиска гиперпараметров для каждого алгоритма.

  Args:
      regressors (dict): Словарь с используемыми алгоритмами регрессии.

  Returns:
      dict: Словарь, где ключи - названия алгоритмов, а значения - словари с диапазонами поиска гиперпараметров.
  """
    search_spaces = {
        "lgbm": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": (4, 16),
            "min_child_samples": (10, 20),
            "learning_rate": [0.01, 0.05, 0.1],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "boosting_type": ["gbdt", "dart"],
            "num_leaves": (32, 128),
        },
        "xgb": {
            "alpha": [0, 0.001],
            "lambda": [0, 0.1],
            "eta": (0.001, 0.05),
            "booster": ["gbtree", "gblinear", "dart"],
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": (4, 16),
            "learning_rate": [0.01, 0.05, 0.1],
            "gamma": [0, 0.1],
            "subsample": [0.6, 0.8],
        },
        "rf": {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": (4, 32),
            "min_samples_split": [2, 4, 8, 16, 24, 32],
            "min_samples_leaf": [1, 2, 4, 8, 16, 24, 32],
        },
        "svm": {
            # "C": [0.01, 0.1, 1, 10, 100],
            'C':(0.1, 10.0),
            "kernel": ['linear', 'rbf', 'sigmoid'],
            "gamma": (0.0001, 1.0),
            'epsilon': [0.01, 0.1, 0.5],
            'max_iter': (1000,10000)
        },
        "ridge": {
          "alpha": (1, 10.0),
          'solver': ['auto','cholesky','lsqr','sparse_cg','sag','saga'],

        },
        "lasso": {
          "alpha": (1, 10.0),  
          "max_iter": (1000,20000) 
        },
        "elastic_net": {
            "alpha": (1, 10.0),  
            "l1_ratio": (0.1, 1.0),  
            "max_iter": (1000, 20000)  
        }
    }
    return search_spaces
"""
  Пример использования define_regressor, define search_space:
  regressors = define_regressors()
  search_spaces = define_search_spaces(regressors)
  """

def evaluate_model(model, X_train, y_train, cv, scoring):
  """
  Оценка модели с помощью кросс-валидации.

  Args:
      model (object): Объект класса алгоритма регрессии.
      X_train (pandas.DataFrame): Обучающая выборка features.
      y_train (pandas.Series): Обучающая выборка target variable.
      cv (int): Число фолдов кросс-валидации.
      scoring (str): Метрика оценки модели (например, 'neg_mean_squared_error').

  Returns:
      float: Среднее значение метрики оценки модели по фолдам кросс-валидации.
  """
  pipe = Pipeline([("model", model)])
  if issubclass(model.__class__, (Lasso, Ridge, SVR,ElasticNet)):
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
  scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
  return -np.mean(scores)  


def run_hyperparameter_tuning(
    X, y, test_size, n_splits, n_repeats, scoring, n_features_to_select, search_methods
):
  """
  Подбор гиперпараметров для каждой модели с помощью оптимизации.

  Args:
      X (pandas.DataFrame): Исходный датасет features.
      y (pandas.Series): Target variable.
      test_size (float): Доля тестовой выборки.
      n_splits (int): Число фолдов кросс-валидации.
      n_repeats (int): Число повторов подбора гиперпараметров.
      scoring (str): Метрика оценки модели (например, 'neg_mean_squared_error').
  n_features_to_select (int): Число признаков для выбора с помощью последовательного выбора признаков.
  search_methods (list): Список методов оптимизации (например, ['gp_minimize', 'gbrt_minimize']).

  Returns:
      tuple: Кортеж из трех элементов:
          - best_models (dict): Словарь, где ключи - методы оптимизации, а значения - словари, где ключи - названия алгоритмов, а значения - объекты класса алгоритмов с лучшими гиперпараметрами.
          - experiment_results (list): Список словарей с результатами подбора гиперпараметров.
          - selected_features_mask (np.array): Маска выбранных признаков.
  """
  X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
  rfc = RandomForestRegressor()
  sfs = SequentialFeatureSelector(
      estimator=rfc,scoring=scoring, n_features_to_select=n_features_to_select, direction="forward"
  )
  sfs.fit(X_train, y_train)
  selected_features_mask = sfs.get_support()
  regressors = define_regressors()
  search_spaces = define_search_spaces(regressors)

  best_models = {search_method: {} for search_method in search_methods}
  experiment_results = []

  for regressor_name, regressor in regressors.items():
    for search_method in search_methods:
      objective_function = lambda params: evaluate_model(
          regressor(**dict(zip(search_spaces[regressor_name].keys(), params))),
          X_train.iloc[:, selected_features_mask],
          y_train,
          cv=n_splits,
          scoring=scoring,
      )
      if search_method == "gp_minimize":
        minimizer = gp_minimize
      elif search_method == "gbrt_minimize":
        minimizer = gbrt_minimize
      else:
        raise ValueError("Invalid search_method. Use 'gp_minimize' or 'gbrt_minimize'.")

      best_params = minimizer(
          func=objective_function,
          dimensions=list(search_spaces[regressor_name].values()),
          n_calls=10,
          n_initial_points=10,
          random_state=None,
          verbose=True,
      )
      best_models[search_method][regressor_name] = regressor(
          **dict(zip(search_spaces[regressor_name].keys(), best_params.x))
      )
      experiment_results.append(
          {"model": regressor_name, "search_method": search_method}
      )

  return best_models, experiment_results, selected_features_mask

"""
Пример использования
search_methods = ['gp_minimize']
n_splits = 5  # Number of folds for cross-validation
n_repeats = 3  # Number of repetitions for hyperparameter tuning
scoring = 'neg_mean_squared_error'  # Evaluation metric (e.g., 'neg_mean_squared_error' for MSE)
n_features_to_select =
  """

def evaluate_regressors(
    regressors, X, y, test_size, number_of_iterations, scoring, selected_features_mask
):
  """
  Оценка производительности подобранных моделей.

  Args:
      regressors (dict): Словарь, где ключи - методы оптимизации, а значения - словари, где ключи - названия алгоритмов, а значения - объекты класса алгоритмов с лучшими гиперпараметрами.
      X (pandas.DataFrame): Исходный датасет features.
      y (pandas.Series): Target variable.
      test_size (float): Доля тестовой выборки.
      number_of_iterations (int): Число повторов оценки.
      scoring (str): Метрика оценки модели (например, 'neg_mean_squared_error').
      selected_features_mask (np.array): Маска выбранных признаков.

  Returns:
      pandas.DataFrame: Таблица с результатами оценки моделей.
  """
  feature_names = [
      "Regressor_name",
      "Regressor param",
      "MAE",
      "MSE",
      "r2_score",
      "Corr",
      "Var_MAE",
      "Var_MSE",
      "Var_r2_score",
      "Var_Corr",
  ]
  result_arr = []
  for search_method, regressor_dict in regressors.items():
        for regressor_name, regressor in regressor_dict.items():
            arr_mae = []
            arr_mse = []
            arr_r2 = []
            arr_predict = []
            arr_corr = []

            for i in np.arange(number_of_iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
                selected_X_train = X_train.iloc[:, selected_features_mask]
                selected_X_test = X_test.iloc[:, selected_features_mask]

                if issubclass(regressor.__class__, (Lasso, Ridge, SVR,ElasticNet)):
                    scaler = MinMaxScaler()
                    selected_X_train = scaler.fit_transform(selected_X_train)
                    selected_X_test = scaler.transform(selected_X_test)
                    
                regressor.fit(selected_X_train, y_train)
                predictions = regressor.predict(selected_X_test)
                mae = mean_absolute_error(y_test.values.ravel(), predictions)
                mse = mean_squared_error(y_test.values.ravel(), predictions)
                r2 = r2_score(y_test.values.ravel(), predictions)
                corr = pearsonr(y_test, predictions)[0]

                arr_mae.append(mae)
                arr_mse.append(mse)
                arr_r2.append(r2)
                arr_corr.append(corr)

           
            mean_MAE = round(np.array(arr_mae).mean(), 3)
            mean_MSE = round(np.array(arr_mse).mean(), 3)
            mean_r2_score = round(np.array(arr_r2).mean(), 3)
            mean_corr = round(np.array(arr_corr).mean(), 5)
            var_MAE = round(stat.variance(np.array(arr_mae)), 3)
            var_MSE = round(stat.variance(np.array(arr_mse)), 3)
            var_r2_score = round(stat.variance(np.array(arr_r2)), 3)
            var_corr = round(stat.variance(np.array(arr_corr)), 5)

            result_arr.append([
                regressor_name,
                str(regressor.get_params()),
                mean_MAE,
                mean_MSE,
                mean_r2_score,
                mean_corr,
                var_MAE,
                var_MSE,
                var_r2_score,
                var_corr,
            ])

  df_results = pd.DataFrame(result_arr, columns=feature_names)
  return df_results


