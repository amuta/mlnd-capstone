import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold




def test_models(X_train, y_train, models, score, seed):
    """
    Computa as validações cruzadas de todos os modelos inputados
    """
    n_folds = 4
    results = []
    names = []
    
    for name, model in models:
        # separa os folds
        folds = StratifiedKFold(n_splits=n_folds, random_state=seed)
        # calcula as cvs
        cv_results = cross_val_score(model, X_train, y_train, cv=folds, scoring=score)
        
        # salva/printa os resultados
        results.append(cv_results)
        names.append(name)
        print('{}\t:{}'.format(name, cv_results.mean(), cv_results.std()))
        
    return names, results


