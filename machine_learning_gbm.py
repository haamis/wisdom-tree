import json, sys
from random import shuffle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.utils import validation

MODE = 'asd'

if MODE == 'cv':

    # params = {
    #     'num_leaves': [15],
    #     'learning_rate': [0.06],
    #     'extra_trees': [True],
    #     'path_smooth': [100000],
    #     'feature_fraction': [1.0],
    #     'cat_smooth': [1.0],
    #     'n_estimators': [10000],
    #     'max_bin': [4,5,6,7,8,9,10,11,12,13,14,15]
    # }

    # params = {'n_estimators': [x for x in range(1000, 9000, 1000)]}
    # params = {'C': [2**x for x in range(-40, 1, 1)], 'class_weight': [None, 'balanced']}
    params = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    sample_num = input_data.shape[0]
    indices = [-1] * (sample_num-sample_num//10) + [0] * (sample_num//10)
    shuffle(indices)
    ps = PredefinedSplit(test_fold=indices)

    # input_data = input_data.toarray().astype('uint8')
    print(input_data.shape)

    input_data = input_data.toarray()#.astype('int16')
    num_categories = np.amax(input_data, axis=0).astype('int16')
    num_categories += 1

    # clf = lgb.LGBMClassifier(objective='binary', n_jobs=6)
    # clf = AdaBoostClassifier() # Test this and non-linear SVCs
    # clf = LinearSVC(dual=False, max_iter=100000, tol=1e-7)
    clf = CategoricalNB(min_categories=num_categories)
    gcv = GridSearchCV(clf, params, verbose=4, cv=ps, return_train_score=True, scoring='accuracy', n_jobs=2)
    # gcv.fit(input_data, winner_data, categorical_feature=categorical_features, feature_name=feature_labels)
    gcv.fit(input_data, winner_data)

    print(gcv.best_params_)
    print(gcv.best_score_)
    print(gcv.cv_results_)

    with open('gcv', 'bw') as out_f:
        import pickle
        pickle.dump(gcv, out_f)

    lgb.plot_importance(gcv.best_estimator_, figsize=(20,35), importance_type='gain', max_num_features=100)
    plt.savefig("importance.svg")
    lgb.plot_tree(gcv.best_estimator_, figsize=(80,80))
    plt.savefig("tree.svg")

else:

    train_data = lgb.Dataset('train.bin')
    validation_data = lgb.Dataset('val.bin', reference=train_data)
    
    # Load num_categories if necessary.
    with open('num_categories.jsonl') as in_f:
        num_categories = json.loads(in_f.read())
    
    param = {}
    param['max_bin_by_feature'] = num_categories
    param['num_leaves'] = 63
    param['objective'] = 'binary'
    param['metric'] = ['binary_error', 'binary']#, 'binary_error', 'l2']
    param['learning_rate'] = 0.05 # 0.06
    # param['boosting'] = 'dart'
    # param['drop_rate'] = 0.1
    # param['max_drop'] = -1
    # param['is_unbalance'] = True
    param['extra_trees'] = True
    param['path_smooth'] = 100_000
    param['feature_fraction'] = 1.0
    # param['bagging_fraction'] = 0.5
    # param['bagging_freq'] = 1
    param['cat_smooth'] = 1.0 # don't smooth categorical variables
    param['min_data_in_leaf'] = 2
    param['max_depth'] = 4
    # param['max_bin'] = 7
    param['num_threads'] = 6
    # param['bin_construct_sample_cnt'] = 1000000
    # param['min_data_in_bin'] = 10
    param['lambda_l1'] = 0.01
    param['lambda_l2'] = 0.01
    # param['early_stopping_round'] = 2000
    param['first_metric_only'] = True

    num_round = 5000
    bst = lgb.train(param, train_data, num_round, valid_names=['val', 'train'], valid_sets=[validation_data, train_data])

    lgb.plot_importance(bst, figsize=(20,35), importance_type='gain', max_num_features=100)
    plt.savefig("importance.svg")
    
    lgb.plot_tree(bst, figsize=(80,80))
    plt.savefig("tree.svg")

    print(param, bst.best_score, num_round, '', sep='\n', file=open('scores.txt', 'at'))