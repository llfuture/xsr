import pandas as pd
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
#if filter the data with go
need_filterwithgo = 0
#if use the pca to prune the data  效果不好
need_usepca = 0
# standard the data  效果不明显
need_data_standard = 0
# fvalue筛选
need_fvalue_filter = 0
fvalue_thres = 10
# 数据增强with SMOTE
need_smote_augument = 0
oversample_rate = 0.4
# 数据增强with boaderline_SMOTE
need_bsmote_augument = 1
# 数据增强with ADASYN
need_adasyn_augument = 0
# 数据增强with Kmeans_SMOTE
need_ksmote_augument = 0

store_oversample_data = 1
# 数据下采样
need_undersampleing = 1
undersample_rate = 0.8
undersample_randomstate = 10
store_undersample_data = 1
# only go data
only_go = 0
# only ppi data
only_ppi = 0

#need_print
need_print = 0
all_columns = ["data","method","data_shape",
               "fvalue_thres",
               "x_train_shape","y_train_shape","x_test_shape","y_test_shape",   # 原始数据的shape
               "one_class_ratio_train","one_class_ratio_test",             # 原始数据中 positive 类在train和test占得比重
               "x_fstat_shape",                 # feature selection 后x的shape
               "one_class_ratio_train_undersample",  "one_class_ratio_train_oversample",          # 采样后positive类在train中占得比重
               "accuracy", "precision",
               "recall", "specificity",
               # "fpr",
               "gmean",
               # "f1"
               ]
all_result = pd.DataFrame(columns=all_columns)
best_result = pd.DataFrame(columns=all_columns)

run_method = "XGBC"

tuning='bay'

dirname = "data/ppi/"
#%%

def nprint(need_print, *arg, **argv):
    if need_print:
        print(*arg, **argv)