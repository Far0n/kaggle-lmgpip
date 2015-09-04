import xgboost as xgb
import numpy as np
from gini_score import *

class PropertyInspectionXGBRegressor(xgb.XGBRegressor):
    def __init__(self, take_log=False, take_sqrt=False, div69=False, take_pow2=False, take_root3=False, eval_set_size=4000, early_stopping_rounds=120,
                 max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1,
                 base_score=0.5, seed=0, missing=None):
        self.take_log = take_log
        self.take_sqrt = take_sqrt
        self.div69 = div69
        self.take_root3 = take_root3
        self.take_pow2 = take_pow2
        self.logistic = True if 'logistic' in objective else False
        self.eval_set_size = eval_set_size
        self.early_stopping_rounds = early_stopping_rounds
        super(xgb.XGBRegressor, self).__init__(max_depth, learning_rate,
                                               n_estimators, silent, objective,
                                               nthread, gamma, min_child_weight,
                                               max_delta_step, subsample,
                                               colsample_bytree,
                                               base_score, seed, missing)

    def fit(self, X, y, eval_set=None):

        X = X.copy()
        y = y.copy()

        if self.take_pow2:
            y = np.power(y,2)
            if eval_set is not None:
                eval_set = [(eval_set[0][0],np.power(eval_set[0][1],2))]

        if self.take_log:
            y = np.log(y)
            if eval_set is not None:
                eval_set = [(eval_set[0][0],np.log(eval_set[0][1]))]

        if self.take_sqrt:
            y = np.sqrt(y)
            if eval_set is not None:
                eval_set = [(eval_set[0][0],np.sqrt(eval_set[0][1]))]

        if self.div69:
            y = y / 69.
            if eval_set is not None:
                eval_set = [(eval_set[0][0], eval_set[0][1] / 69.)]

        if self.take_root3:
            y = np.power(y,1./3.)
            if eval_set is not None:
                eval_set = [(eval_set[0][0],np.power(eval_set[0][1],1./3.))]

        if self.logistic:
            y = (y - min(y)) / (1.0 * max(y) - min(y))

        offset = self.eval_set_size

        if eval_set is None:
            eval_set = [(X[:offset, :], y[:offset])]
            X = X[offset:, :]
            y = y[offset:]

        if self.take_log:
            _eval_metric = xgb_feval_gini_log
        elif self.take_root3:
            _eval_metric = xgb_feval_gini_root3
        elif self.take_sqrt:
            _eval_metric = xgb_feval_gini_sqrt
        elif self.take_pow2:
            _eval_metric = xgb_feval_gini_pow2
        elif self.div69:
            _eval_metric = xgb_feval_gini_div69
        else:
            _eval_metric = xgb_feval_gini

        super(xgb.XGBRegressor, self).fit(X, y, eval_set=eval_set, eval_metric=_eval_metric,
                                           early_stopping_rounds=self.early_stopping_rounds, verbose=0)

        return self

    def fit_cv(self, X, y, eval_set=None):
        return self.fit(X, y, eval_set)

    def predict(self, data):
        test_dmatrix = xgb.DMatrix(data, missing=self.missing)

        if hasattr(self, 'best_iteration'):
            pred = self.booster().predict(test_dmatrix, ntree_limit=self.best_iteration)
        else:
            pred = self.booster().predict(test_dmatrix)

        if self.take_log:
            return np.exp(pred)
        elif self.take_root3:
           return np.power(pred,3)
        elif self.take_sqrt:
            return np.power(pred,2)
        elif self.take_pow2:
            return np.sqrt(pred)
        elif self.div69:
            return pred * 69
        else:
            return pred