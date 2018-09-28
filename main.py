from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
DROP_LIST = ['Id', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']
CV = 10
input_dim = 30


def scaling(df, mean):
    df = df.drop(DROP_LIST, axis=1)
    df.fillna(mean, inplace=True)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df = MinMaxScaler().fit_transform(df)
    return df


def random_feature(X, y):
    rf = RandomForestRegressor(n_estimators=80, max_features='auto')
    rf.fit(X, y)
    ranking = np.argsort(rf.feature_importances_)
    return ranking


def build_model():
    sequential = Sequential()
    sequential.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    sequential.add(Dense(16, kernel_initializer='normal', activation='relu'))
    sequential.add(Dense(1, kernel_initializer='normal'))
    sequential.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['mse'])
    return sequential


def reg_nn(x_train, y_train, x_test):
    estimator = KerasRegressor(build_fn=build_model, epochs=10, batch_size=16, verbose=1, validation_split=0.2)
    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(x_test, batch_size=16, verbose=1)
    return y_pred


def reg_svr(x_train, y_train, x_test):
    parameters = {'kernel': ['rbf'], 'gamma': np.logspace(-2, 2, 5), 'C': [1e0, 1e1, 1e2, 1e3]}
    svr = SVR()
    cv = GridSearchCV(svr, parameters, n_jobs=-1, return_train_score=False,
                      cv=CV, scoring='neg_mean_squared_error', verbose=3)
    cv.fit(x_train, y_train)

    print('Best Score:', cv.best_score_)
    print('Best Estimator:\n', cv.best_estimator_)

    y_pred = cv.predict(x_test)
    return y_pred


def reg_xgb(x_train, y_train, x_test):
    model = xgb.XGBRegressor()
    param = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}
    cv = GridSearchCV(model, param, verbose=3, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')
    cv.fit(x_train, y_train)

    print('Best Score:', cv.best_score_)
    print('Best Estimator:\n', cv.best_estimator_)
    y_pred = cv.predict(x_test)
    return y_pred


def main():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # print(df_train.info())
    # print(df_test.info())

    for i in range(df_train.shape[1]):
        if df_train.iloc[:, i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(df_train.iloc[:, i].values) + list(df_test.iloc[:, i].values))
            df_train.iloc[:, i] = lbl.transform(list(df_train.iloc[:, i].values))
            df_test.iloc[:, i] = lbl.transform(list(df_test.iloc[:, i].values))

    x_train = df_train.drop('SalePrice', axis=1)
    x_test = df_test
    xMat = pd.concat([x_train, x_test])
    x_train = scaling(x_train, xMat.mean())
    x_test = scaling(x_test, xMat.mean())
    y_train = df_train['SalePrice']
    # y_train = np.log(y_train)

    ranking = random_feature(x_train, y_train)
    x_train = x_train[:, ranking[:input_dim]]
    x_test = x_test[:, ranking[:input_dim]]

    y_pred = reg_xgb(x_train, y_train, x_test)

    df_pred = pd.DataFrame(y_pred, columns=['SalePrice'])
    df_test['SalePrice'] = df_pred['SalePrice']
    df_test[['Id', 'SalePrice']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)


if __name__ == '__main__':
    main()
