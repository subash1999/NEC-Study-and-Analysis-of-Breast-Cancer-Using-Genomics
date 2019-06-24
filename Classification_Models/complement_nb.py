import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB

from naives_bayes_parent import GenomicsNB

class GenomicsCNB(GenomicsNB) :
     
    def __init__(self):
        super().__init__()

    def classification(self,x_train : np.array ,y_train : np.array) -> ComplementNB() :        
        """This returns the ComplementNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            ComplementNB -- It is a KNeighborsClassifier model
        """
        clf = ComplementNB()
        clf.fit(x_train,y_train)        
        return clf
