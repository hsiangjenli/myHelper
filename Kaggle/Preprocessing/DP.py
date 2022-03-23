import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()


class DataPreprocessing:
    def __init__(self):
        pass
    def fit(self, df, target, printf=True):
        self.df = df
        self.target = target
        self.printf = printf
        self.auto_selection()
        self.remove_target()
        if printf:
            self.print_info()
        
    def print_info(self):
        print(f'● Numeric columns:\n {self.num_col}')
        print(f'● Categorical columns:\n  -unordered:\n  {self.unord_cat_col}\n  -ordered:\n  {self.ord_cat_col}')
        print(f'● Binary columns:\n {[[col, list(self.df[col].unique())] for col in self.binary_col]}')
        
    def auto_selection(self):
        self.binary_col = [col for col in self.df.columns if len(self.df[col].unique()) == 2]
        self.num_col = [col for col in self.df.select_dtypes(include=np.number).columns if col not in self.binary_col]
        self.unord_cat_col = [col for col in self.df.select_dtypes(include='object').columns if col not in self.binary_col]
        self.ord_cat_col = list()
        return self
    
    def remove_target(self):
        columns = [self.binary_col, self.num_col, self.unord_cat_col, self.ord_cat_col]
        for col in columns:
            if self.target in col:
                col.remove(self.target)
        return self
    
    def transform(self, scalar=SS, test_size=0.3, random_state=1):
        #self.split_data(test_size=test_size, random_state=random_state)
        self.transform_onehotenc(test_size=test_size, random_state=random_state)
        self.transform_scaler(scalar=scalar, test_size=test_size, random_state=random_state)
        X_train = np.append(self.X_train_cat, self.X_train_num, axis=1)
        X_test = np.append(self.X_test_cat, self.X_test_num, axis=1)
        
        y_train, y_test = train_test_split(self.df[self.target], 
                                           test_size=test_size, 
                                           random_state=random_state)
        return X_train, X_test, y_train, y_test
        
    def transform_scaler(self, scalar, test_size, random_state):
        X_train, X_test = train_test_split(self.df[self.num_col],
                                           test_size=test_size, 
                                           random_state=random_state)
        self.scalar = scalar
        self.scalar.fit(X_train)
        self.X_train_num = self.scalar.transform(X_train)
        self.X_test_num = self.scalar.transform(X_test)
        return self
    def transform_onehotenc(self, test_size, random_state):
        cat_col = self.unord_cat_col+self.binary_col
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(self.df[cat_col])
        transformed_data = self.enc.transform(self.df[cat_col])
        self.X_train_cat, self.X_test_cat = train_test_split(transformed_data,
                                                             test_size=test_size,
                                                             random_state=random_state)
        #transformed_feature_names = list(enc.get_feature_names_out(cat_col))
        #self.df[transformed_feature_names] = enc.transform(self.df[cat_col])
        #self.df = self.df.drop(columns=cat_col)
        return self