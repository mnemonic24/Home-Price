from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from datetime import datetime

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
DROP_LIST = ['Id', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']
CV = 10


def scaling(df):
    df = df.drop(DROP_LIST, axis=1)
    df.fillna(df.mean(), inplace=True)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df = MinMaxScaler().fit_transform(df)
    return df


def main():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    print(df_train.info())
    print(df_test.info())

    for i in range(df_train.shape[1]):
        if df_train.iloc[:, i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(df_train.iloc[:, i].values) + list(df_test.iloc[:, i].values))
            df_train.iloc[:, i] = lbl.transform(list(df_train.iloc[:, i].values))
            df_test.iloc[:, i] = lbl.transform(list(df_test.iloc[:, i].values))

    x_train = scaling(df_train.drop('SalePrice', axis=1))
    x_test = scaling(df_test)
    y_train = df_train['SalePrice']

    parameters = {"n_estimators": [3, 10, 100, 1000],
                  "bootstrap": [True, False],
                  "n_jobs": [-1]}

    rf = RandomForestRegressor()
    cv = GridSearchCV(rf, parameters, n_jobs=-1, return_train_score=False,
                      cv=CV, scoring='neg_mean_squared_error', verbose=3)
    cv.fit(x_train, y_train)

    print('Best Score:', cv.best_score_)
    print('Best Estimator:\n', cv.best_estimator_)

    test_pred = cv.predict(x_test)
    df_pred = pd.DataFrame(test_pred, columns=['SalePrice'])
    df_test['SalePrice'] = df_pred['SalePrice']
    df_test[['Id', 'SalePrice']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)


if __name__ == '__main__':
    main()
