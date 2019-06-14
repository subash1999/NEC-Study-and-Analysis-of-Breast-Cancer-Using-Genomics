import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

from project3_parent import Project3Parent


class GenomicsLR(Project3Parent):
    """This performs the linear regression on our data
    
    ATTRIBUTE
    ---------
    
    FUNCTION
    --------

    """
    # def init method , i.e. constructor 
    def __init__(self):
        super().__init__()
    
    def now(self):
        df = self.getDF()
        print(df.head())
        df = df.drop(columns= ['GEO_ACC'] )
        y = np.array(df['relapse'])
        x = np.array(df.drop(columns= ['relapse']))
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)
        clf = LinearRegression(n_jobs=-1)
        clf.fit(x_train,y_train)
        print(clf.coef_)
        
        acc_train = clf.score(x_train,y_train)
        y_out = clf.predict(x_test)
        y_res = []
        for y in y_out :
            if y >=0.5 :
                y_res.append(1)
            else : 
                y_res.append(0)
        acc_test = clf.score(x_test,y_res) 

        print('acc_train : '+str(acc_train),'\nacc_test : '+str(acc_test))
    
g = GenomicsLR()
g.now()
