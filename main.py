from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pandas_profiling
from datetime import datetime
from sklearn.metrics import mean_squared_error, make_scorer

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'


def main():
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    # pandas_profiling.ProfileReport(df_train)
    # pandas_profiling.ProfileReport(df_test)

    for i in range(df_train.shape[1]):
        if df_train.iloc[:, i].dtypes == object:
            lbl = LabelEncoder()
            lbl.fit(list(df_train.iloc[:, i].values) + list(df_test.iloc[:, i].values))
            df_train.iloc[:, i] = lbl.transform(list(df_train.iloc[:, i].values))
            df_test.iloc[:, i] = lbl.transform(list(df_test.iloc[:, i].values))

    x_train = df_train.drop(['Id', 'SalePrice'], axis=1).fillna(df_train.mean())
    y_train = df_train['SalePrice']
    x_test = df_test.drop('Id', axis=1).fillna(df_test.mean())

    # Xmat = pd.concat([x_train, x_test])
    # sns.distplot(y_train)
    # plot.show()

    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)

    parameters = {'n_estimators': [10, 100], 'n_jobs': [-1]}
    rf = RandomForestRegressor()
    cv = GridSearchCV(rf, parameters, n_jobs=-1, return_train_score=False, cv=5, scoring='mean_squared_error', verbose=10)
    cv.fit(x_train, y_train)

    print('Best Score:', cv.best_score_)
    print('Best Estimator:\n', cv.best_estimator_)

    test_pred = cv.predict(x_test)
    df_pred = pd.DataFrame(test_pred, columns=['SalePrice'])
    df_test['SalePrice'] = df_pred['SalePrice']
    df_test[['Id', 'SalePrice']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)


if __name__ == '__main__':
    main()
