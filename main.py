# Zarządzanie bibliotekami - import wszystkich potrzebnych bibliotek do dalszej analizy

import pandas as pd
from pandas import read_csv

import numpy as np

import matplotlib
import matplotlib. pyplot as plt
import matplotlib. dates as mandates
from matplotlib import pyplot

import xgboost as xg
import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn. preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from keras. layers import LSTM, Dense, Dropout
from keras. models import Sequential, load_model
import keras. backend as K
from keras. callbacks import EarlyStopping
from keras. utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor


# Funkcja importująca zbiór danych i dodająca RoRy

def import_explore(dataset):
    series = read_csv(dataset)  # , parse_dates=['date']), index_col='date')

    series['RoR_date'] = (series.groupby('symbol')['adj_close_date'].apply(
        pd.Series.pct_change) + 1)
    series['RoR_mtd'] = (series.groupby('symbol')['adj_close_mtd'].apply(
        pd.Series.pct_change) + 1)
    series['RoR_qtd'] = (series.groupby('symbol')['adj_close_qtd'].apply(
        pd.Series.pct_change) + 1)
    series['RoR_htd'] = (series.groupby('symbol')['adj_close_htd'].apply(
        pd.Series.pct_change) + 1)
    series['RoR_ytd'] = (series.groupby('symbol')['adj_close_ytd'].apply(
        pd.Series.pct_change) + 1)

    df = series.dropna()

    return (df)

    def getX(dataset, X_names):
        X = pd.DataFrame(dataset[X_names])
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        return (X)


def getY(dataset, Y_name):
    Y = pd.DataFrame(dataset[Y_name])
    scaler = MinMaxScaler(feature_range=(0, 1))
    Y = scaler.fit_transform(Y)
    return (Y)

# X = getX(df, X_col)
# Wybór zmiennej objaśnianej (horyzontu czasowego analizy)
# Y_name = Y_col[1]
# Y = getY(df, Y_name)

# Regresja liniowa wersja 3

def linear_regression_3(X1, X2, y1, y2):
    model = linear_model.LinearRegression()
    X_train = pd.DataFrame(X1)
    y_train = pd.DataFrame(y1)

    model.fit(X_train, y_train)

    Y_pred_train = model.predict(X_train)

    rmse = np.sqrt(mean_squared_error(y_train, Y_pred_train))
    print("RMSE dla regresji 3 (zbiór uczący się): % f" % (rmse))

    X_test = pd.DataFrame(X2)
    Y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y2, Y_pred_test))
    print("RMSE dla regresji 3 (zbiór testowy): % f" % (rmse))
    print("Wyniki walidacji krzyżowej dla regresji liniowej:")
    linear_scores = cross_validate(model, X, Y,
                                   scoring="neg_root_mean_squared_error")
    print(linear_scores)
    for k, v in linear_scores.items():
        print(k, v.mean())

# linear_regression_3(X_train, X_test, y_train, y_test)

# Model XGBoosting

def XGBmodel(X1, X2, y1, y2):
    train_dmatrix = xg.DMatrix(data=X1, label=y1)
    test_dmatrix = xg.DMatrix(data=X2, label=y2)

    param = {"booster": "gblinear", "objective": "reg:linear"}

    xgb_r = xg.train(params=param, dtrain=train_dmatrix, num_boost_round=10)
    pred = xgb_r.predict(test_dmatrix)

    rmse = np.sqrt(mean_squared_error(y2, pred))
    print("")
    print("RMSE dla XGBoosting: % f" % (rmse))
    print("Wyniki walidacji krzyżowej dla XGBoost:")
    data_dmatrix = xg.DMatrix(data=X_train, label=y_train)
    xgb_cv = xg.cv(dtrain=data_dmatrix, params=param, nfold=5, metrics='rmse',
                   seed=42)
    print(xgb_cv["train-rmse-mean"].mean())
    print("")
# XGBmodel(X_train, X_test, y_train, y_test)

# Model LSTM

def LSTMmodel(X1, X2, y1, y2):
    model = Sequential()
    model.add(
        LSTM(units=50, return_sequences=True, input_shape=(X1.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X1, y1, epochs=10, batch_size=32)  # Przy testach 100 zmienić na 1
    pred = model.predict(X2)
    rmse = np.sqrt(mean_squared_error(y2, pred))
    print("")
    print("RMSE dla LSTM: % f" % (rmse))

# LSTMmodel(X_train, X_test, y_train, y_test)

# Model LightGBM

def LGBMmodel(X1, X2, y1, y2):
      lgbm = lgb.LGBMRegressor()

      lgbm.fit(X1, y1)
      pred = lgbm.predict(X2)

      rmse = np.sqrt(mean_squared_error(y2, pred))
      print("RMSE dla LGBM: % f" %(rmse))
      print("")
      print("Wyniki walidacji krzyżowej dla LightGBM:")
      lgbm_scores = cross_validate(lgbm, X, Y, scoring="neg_root_mean_squared_error")
      print(lgbm_scores)
      for k, v in lgbm_scores.items():
          print(k, v.mean())

#LGBMmodel(X_train, X_test, y_train, y_test)

# KOD - Komórka do uzyskania wyników (wcześniej zaimportuj biblioteki i wywołaj funkcje)

#Funkcja 1
data = 'convictions_returns.csv'
df = import_explore(data)

# Porządkowanie kolumn w zbiorze danych

# Wektor nazw zmiennych objaśnianych Y
Y_col = ['RoR_date','RoR_mtd','RoR_qtd','RoR_htd','RoR_ytd']
# Wektor nazw kolumn do usunięcia
Others = ['Unnamed: 0','symbol','sector','date']
all_column = df.columns
X_col = np.setdiff1d(all_column, Y_col)
# Wektor nazw zmeinnych objaśniających X
X_col = np.setdiff1d(X_col, Others)

# Funkcja 2 i 3
X = getX(df, X_col)
#Wybór zmiennej objaśnianej (horyzontu czasowego analizy)
check_list = ['RoR_date', 'RoR_mtd', 'RoR_qtd', 'Ror_htd', 'RoR_ytd']
Y_name = input("Wprowadź interesujący cię zakres: RoR_date, RoR_mtd, RoR_qtd, Ror_htd, RoR_ytd ")
while Y_name not in check_list:
    Y_name = input("Wprowadź poprawny zakres: RoR_date, RoR_mtd, RoR_qtd, Ror_htd, RoR_ytd ")
    if Y_name in check_list:
        break
Y = getY(df, Y_name)

# Podział zbioru na uczący się i testowy
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 23, shuffle = True)

# Funkcja 4
linear_regression_3(X_train, X_test, y_train, y_test)
# Funkcja 5
XGBmodel(X_train, X_test, y_train, y_test)
# Funkcja 6
LSTMmodel(X_train, X_test, y_train, y_test)
# Funkcja 7
LGBMmodel(X_train, X_test, y_train, y_test)