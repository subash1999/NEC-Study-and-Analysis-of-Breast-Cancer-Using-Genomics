import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from naives_bayes_parent import GenomicsNB

class GenomicsGNB(GenomicsNB) :
     
    def __init__(self):
        super().__init__()

    def classification(self,x_train : np.array ,y_train : np.array) -> GaussianNB() :        
        """This returns the GaussianNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            GaussianNB -- It is a KNeighborsClassifier model
        """
        clf = GaussianNB()
        clf.fit(x_train,y_train)        
        return clf
