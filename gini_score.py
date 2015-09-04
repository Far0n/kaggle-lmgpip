import numpy as np


def gini(y_true, y_pred, cmpcol=0, sortcol=1):
    assert (len(y_true) == len(y_pred))
    all = np.asarray(np.c_[y_true, y_pred, np.arange(len(y_true))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(y_true) + 1) / 2.
    return giniSum / len(y_true)


def gini_normalized(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


def xgb_feval_gini(preds, dtrain):
    predictions = preds
    labels = dtrain.get_label()
    return 'gini ', -gini_normalized(labels, predictions)


def xgb_feval_gini_log(preds, dtrain):
    predictions = np.exp(preds)
    labels = np.exp(dtrain.get_label())
    return 'gini ', -gini_normalized(labels, predictions)


def xgb_feval_gini_root3(preds, dtrain):
    predictions = np.power(preds, 3.)
    labels = np.power(dtrain.get_label(), 3)
    return 'gini ', -gini_normalized(labels, predictions)


def xgb_feval_gini_sqrt(preds, dtrain):
    predictions = np.power(preds, 2)
    labels = np.power(dtrain.get_label(), 2)
    return 'gini ', -gini_normalized(labels, predictions)


def xgb_feval_gini_div69(preds, dtrain):
    predictions = preds * 69.
    labels = 69. * dtrain.get_label()
    return 'gini ', -gini_normalized(labels, predictions)


def xgb_feval_gini_pow2(preds, dtrain):
    predictions = np.sqrt(preds)
    labels = np.sqrt(dtrain.get_label())
    return 'gini ', -gini_normalized(labels, predictions)
