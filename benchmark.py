import numpy as np
import lightgbm as lgb
data = np.random.rand(90000, 113)  # 500 entities, each contains 10 features
second_data = np.zeros((90000,3000))
data = np.hstack((data,second_data))
label = np.random.randint(2, size=90000)  # binary target
train_data = lgb.Dataset(data, label=label)
# param = {'num_leaves': 31, 'objective': 'binary'}#, 'device': 'gpu'}
# param['metric'] = 'auc'
param = {}#'device': 'gpu'}
param['num_leaves'] = 5
param['objective'] = 'binary'
param['metric'] = ['average_precision', 'binary']#, 'binary_error', 'l2']
param['learning_rate'] = 0.1
# param['boosting'] = 'dart'
# param['drop_rate'] = 0.9
# param['max_drop'] = -1
# param['is_unbalance'] = True
param['extra_trees'] = True
param['path_smooth'] = 1_000_000
param['feature_fraction'] = 0.1
param['cat_smooth'] = 0.0 # don't smooth categorical variables
# param['min_data_in_leaf'] = 2
# param['max_depth'] = 5
# param['max_bin'] = 8
# param['num_threads'] = 6
# param['bin_construct_sample_cnt'] = 100000
# param['min_data_in_bin'] = 10000

num_round = 10000
# num_round = 1000
bst = lgb.train(param, train_data, num_round, valid_sets=[train_data], categorical_feature=[0] + [x for x in range(25,3000)])