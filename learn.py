import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn import metrics
from sklearn.linear_model import LassoCV
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
from gini_score import *
from sklearn.externals import joblib
from xgbex import PropertyInspectionXGBRegressor

# This solution is based on:
# "Bench-Stacked-Generalization" (https://www.kaggle.com/justfor/liberty-mutual-group-property-inspection-prediction/bench-stacked-generalization)
# with ideas taken from
# "Blah-XGB" (https://www.kaggle.com/soutik/liberty-mutual-group-property-inspection-prediction/blah-xgb)

seed = 42
nthread = 2
silent = 1
xgb_n_estimators = 10000
n_folds = 12
esr = 360


def get_ranks(x):
    ind = x.argsort()
    ranks = np.empty(len(x), int)
    ranks[ind] = np.arange(len(x))
    return ranks


def get_data(training_file, test_file):
    drop_out = ['T1_V10', 'T1_V13', 'T2_V7', 'T2_V10']

    train = pd.read_csv('../input/' + training_file)
    test = pd.read_csv('../input/' + test_file)

    features = list(train.columns[2:])
    features = np.setdiff1d(features, drop_out)

    y_train = np.array(train.Hazard)
    y_test = None

    train_ids = np.array(train.Id)
    test_ids = np.array(test.Id)

    x_train = np.array(train[features].astype(float))
    x_test = np.array(test[features].astype(float))

    return x_train, y_train, x_test, y_test, train_ids, test_ids


xgb_params_logistic = {
    "objective": "reg:logistic",
    "learning_rate": 0.005,
    "max_depth": 9,
    "min_child_weight": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "silent": silent,
    "nthread": nthread
}

xgb_params_linear = {
    "objective": "reg:linear",
    "learning_rate": 0.005,
    "max_depth": 9,
    "min_child_weight": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "silent": silent,
    "nthread": nthread
}

xgb_params_poisson = {
    "objective": "count:poisson",
    "learning_rate": 0.005,
    "max_depth": 9,
    "min_child_weight": 6,
    "subsample": 0.7,
    "max_delta_step": 0.7,
    "colsample_bytree": 0.7,
    "silent": silent,
    "nthread": nthread
}

if __name__ == '__main__':

    np.random.seed(seed)

    x_train2, y_train2, x_test2, y_test2, train_ids, test_ids = get_data("train2.csv", "test2.csv")
    x_train3, y_train3, x_test3, y_test3, train_ids, test_ids = get_data("train3.csv", "test3.csv")
    x_train4, y_train4, x_test4, y_test4, train_ids, test_ids = get_data("train4.csv", "test4.csv")

    DATA2 = (x_train2, y_train2, x_test2, y_test2)
    DATA3 = (x_train3, y_train3, x_test3, y_test3)
    DATA4 = (x_train4, y_train4, x_test4, y_test4)

    clfs = [
        [DATA2, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA2, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA2, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA2, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA2, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],
        [DATA2, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],

        [DATA3, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA3, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA3, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA3, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA3, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],
        [DATA3, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],

        [DATA4, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA4, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_linear)],
        [DATA4, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA4, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_logistic)],
        [DATA4, PropertyInspectionXGBRegressor(take_log=False, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],
        [DATA4, PropertyInspectionXGBRegressor(take_log=True, n_estimators=xgb_n_estimators, early_stopping_rounds=esr,
                                               **xgb_params_poisson)],
    ]

    skf = KFold(n=x_train2.shape[0], n_folds=n_folds)

    blend_train = np.zeros((x_train2.shape[0], len(clfs)))
    blend_test = np.zeros((x_test2.shape[0], len(clfs)))

    start_time = datetime.now()
    cv_results = np.zeros((len(clfs), len(skf)))

    for j, data_clf in enumerate(clfs):
        X_dev = data_clf[0][0]
        Y_dev = data_clf[0][1]
        X_test = data_clf[0][2]
        Y_test = data_clf[0][3]
        clf = data_clf[1]
        print ('\nTraining classifier [%s]: %s' % (j, clf))
        blend_test_j = np.zeros((X_test.shape[0], len(skf)))
        for i, (train_index, cv_index) in enumerate(skf):
            # print ('Fold [%s]' % (i))

            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]

            # print("fit")
            if 'fit_cv' in dir(clf):
                clf.fit_cv(X_train, Y_train, [(X_cv, Y_cv)])
            else:
                clf.fit(X_train, Y_train)

            one_result = clf.predict(X_cv)
            blend_train[cv_index, j] = one_result
            cv_score = gini_normalized(Y_cv, blend_train[cv_index, j])
            cv_results[j, i] = cv_score
            score_mse = metrics.mean_absolute_error(Y_cv, one_result)
            print ('Fold [%s] norm. Gini = %0.5f, MSE = %0.5f' % (i, cv_score, score_mse))
            blend_test_j[:, i] = clf.predict(X_test)
        blend_test[:, j] = blend_test_j.mean(1)
        print ('Clf_%d Mean norm. Gini = %0.5f (%0.5f)' % (j, cv_results[j,].mean(), cv_results[j,].std()))

    end_time = datetime.now()
    time_taken = (end_time - start_time)
    print ("Time taken for pre-blending calculations: {0}".format(time_taken))
    print ("CV-Results", cv_results)
    print ("Blending models.")

    bclf = LassoCV(n_alphas=100, alphas=None, normalize=True, cv=5, fit_intercept=True, max_iter=10000, positive=True)
    bclf.fit(blend_train, Y_dev)

    Y_test_predict = bclf.predict(blend_test)

    cv_score = cv_results.mean()
    print ('Avg. CV-Score = %s' % (cv_score))
    submission = pd.DataFrame({"Id": test_ids, "Hazard": Y_test_predict})
    submission = submission.set_index('Id')
    submission.to_csv("farons_solution.csv")
