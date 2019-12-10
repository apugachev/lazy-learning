import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from progressbar import progressbar as pb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from copy import deepcopy
np.random.seed(4)

class LazyLearner:
    def __init__(self, df, numerical_cols, binary_cols, custom_pivots=None, 
                 seed=42, test_size=0.2, threshold=None):
        self.df = df
        self.train_df = None
        self.test_df = None
        self.target = 'price_range'
        self.numerical_cols = numerical_cols
        self.binary_cols = binary_cols
        self.custom_pivots = custom_pivots
        self.seed = seed
        self.test_size = test_size
        self.threshold = threshold
        
        
    def onehot_numerical(self):
        
        final_df = pd.DataFrame()
        
        for col in self.numerical_cols:
            min_b = self.df[col].min()
            max_b = self.df[col].max()
            diff = (max_b - min_b)
            
            if not self.custom_pivots:
                pivots = [min_b + i * diff // 5 for i in range(1,5)]
            else:
                pivots = self.custom_pivots[col]
                
                if pivots == None:
                    pivots = [min_b + i * diff // 5 for i in range(1,5)]

            categories_inds = np.zeros(len(self.df), dtype=int)
            
            for i in range(len(pivots)+1):
                if i == 0:
                    new_df = self.df[self.df[col] <= pivots[i]]
                elif i == len(pivots):
                    new_df = self.df[self.df[col] >= pivots[i-1]]
                else:
                    new_df = self.df[self.df[col] > pivots[i-1]] 
                    new_df = new_df[new_df[col] <= pivots[i]]
                    
                categories_inds[new_df.index.tolist()] = i 
                
            categories_inds = pd.Series(list(categories_inds))
            df_part = pd.get_dummies(categories_inds, prefix=col)
            
            for new_col in df_part.columns:
                final_df[new_col] = df_part[new_col]
                
        for new_col in self.binary_cols:
            final_df[new_col] = self.df[new_col]
            
        return final_df
            
        
    def calc_supports_on_binaries(self, threshold=None, get_weights=False):
        binarized_df = self.onehot_numerical()
        
        cols = binarized_df.columns.tolist()
        cols_d = {i:cols[i] for i in range(len(cols))}
        
        X_train, X_test = train_test_split(binarized_df, test_size=self.test_size, random_state=self.seed)
            
        self.train_df = X_train
        self.test_df = X_test
        
        if get_weights:
            X_test = deepcopy(X_train)
        
        X_train_np, X_test_np = np.array(X_train), np.array(X_test)
        
        X_train_pos = np.array(X_train[X_train[self.target] == 1])
        X_train_neg = np.array(X_train[X_train[self.target] == 0])

        support_pos, support_neg = [], []

        for i in pb(range(len(X_test))):
            support_pos_i, support_neg_i = [], []
            
            for j in range(len(X_train)):
                test_ones = np.where(X_test_np[i,:-1] == 1)[0]
                train_ones = np.where(X_train_np[j,:-1] == 1)[0]
                both_ones = np.intersect1d(test_ones, train_ones)

                if len(both_ones) > 0:
                    X_sub_pos = deepcopy(X_train_pos)
                    X_sub_neg = deepcopy(X_train_neg)
                    
                    X_sub_pos_slice = X_sub_pos[:,both_ones]
                    X_sub_neg_slice = X_sub_neg[:,both_ones]
                    
                    X_sub_pos_sum = np.sum(X_sub_pos_slice, axis=1)
                    X_sub_neg_sum = np.sum(X_sub_neg_slice, axis=1)
                    
                    X_sub_pos_chosen = X_sub_pos_sum[X_sub_pos_sum == len(both_ones)]
                    X_sub_neg_chosen = X_sub_neg_sum[X_sub_neg_sum == len(both_ones)]
                    
                    support_pos_i.append(len(X_sub_pos_chosen) / len(X_train_pos))
                    support_neg_i.append(len(X_sub_neg_chosen) / len(X_train_neg))
                    
            if self.threshold:
                support_pos_i = np.array(support_pos_i)
                support_neg_i = np.array(support_neg_i)
                support_pos_i = support_pos_i[support_pos_i > self.threshold]
                support_neg_i = support_neg_i[support_neg_i > self.threshold]
                
            support_pos.append(support_pos_i)
            support_neg.append(support_neg_i)
            
        return support_pos, support_neg
    
    def get_logreg(self):
        sup_pos, sup_neg = self.calc_supports_on_binaries(get_weights=True) 
        sup_pos_means = [np.mean(lst) for lst in sup_pos]
        sup_neg_means = [np.mean(lst) for lst in sup_neg]
        
        X_train_w = np.array([sup_pos_means, sup_neg_means]).T
        clf = LogisticRegression()
        clf.fit(X_train_w, self.train_df[self.target])
        
        return clf
    
    def get_preds(self, aggregate='mean'):
        sup_pos, sup_neg = self.calc_supports_on_binaries()   
        y_pred = []
        
        if aggregate == 'mean':
            aggr_func = np.mean
        if aggregate == 'max':
            aggr_func = np.max
        if aggregate == 'median':
            aggr_func = np.median
            
        if aggregate == 'weighted_mean':
            logreg = self.get_logreg()
            sup_pos_means = [np.mean(lst) for lst in sup_pos]
            sup_neg_means = [np.mean(lst) for lst in sup_neg]
            y_pred = logreg.predict(np.array([sup_pos_means, sup_neg_means]).T)
        
        else:
            for pos, neg in zip(sup_pos, sup_neg):

                pos_mean = aggr_func(pos)
                neg_mean = aggr_func(neg)

                if pos_mean > neg_mean:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                
        return y_pred
    
    
    def calc_all_metrics(self, y_true, y_pred):
        
        metrics_dict = {}
        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_pred)
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        metrics_dict['precision'] = precision_score(y_true, y_pred)
        metrics_dict['recall'] = recall_score(y_true, y_pred)
        metrics_dict['f1'] = f1_score(y_true, y_pred)
        
        return metrics_dict
        
    def calc_lazy_metrics(self, aggregate='mean'):
        y_pred = self.get_preds(aggregate)
        y_true = self.test_df[self.target]
        
        result = self.calc_all_metrics(y_true, y_pred)
        
        return result
    
    def calc_logreg_metrics(self, c_val=1.0, penalty='l2'):        
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)
        X_train_np, X_test_np = np.array(X_train), np.array(X_test)
        y_true = X_test_np[:,-1]
       
        clf = LogisticRegression(penalty=penalty, C=c_val, n_jobs=4, random_state=self.seed)
        
        clf.fit(X_train_np[:,:-1], X_train_np[:,-1])
        y_pred = clf.predict(X_test_np[:,:-1])
        
        result = self.calc_all_metrics(y_true, y_pred)
       
        return result       
            
        
    def calc_forest_metrics(self, estim=10, depth=None, leaf=1):
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)
        X_train_np, X_test_np = np.array(X_train), np.array(X_test)
        y_true = X_test_np[:,-1]
        
        clf = RandomForestClassifier(max_depth=depth, min_samples_leaf=leaf, 
                                     n_estimators=estim, random_state=self.seed, n_jobs=4)

        clf.fit(X_train_np[:,:-1], X_train_np[:,-1])
        y_pred = clf.predict(X_test_np[:,:-1])
        
        result = self.calc_all_metrics(y_true, y_pred)
       
        return result  