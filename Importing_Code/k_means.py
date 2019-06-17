import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from project3_parent import Project3Parent

class GenomicsKMeans(Project3Parent):
    
    def __init__(self):
        super().__init__()
    
    def trainModel(self,test_size_input: int =0.2, no_of_cluster = 2) -> KMeans() :
        """This method trains the model with KMeans classification, before it call makeDF()
        
        Keyword Arguments:
            test_size_input {float} -- Fraction of data used for testing (default: {0.2:float})
            number_of_cluster {int} -- number of clusters the data is to be categorized
            
        Returns:
            sklearn.linear_model -- KMeans Model of the given data
        """
        x_train,x_test,y_train,y_test = self.splitDataToTrainTest(test_size_input) 
        clf = self.classification(x_train,y_train, no_of_cluster)
                
        self.model = clf
        print(silhouette_score(x_train, clf.labels_))
        return clf

    def classification(self,x_train : np.array ,y_train : np.array, no_of_cluster = 2) -> KMeans() :        
        """This returns the KMeans Clustering Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            KMeans -- It is a KNeighborsClassifier model
        """
        clf = KMeans(n_clusters = no_of_cluster)
        clf.fit(x_train)
        return clf

    def predict(self,x : np.array) -> np.array :
        """This method predicts the value given the input data, use trainModel() before this method
        
        Arguments:
            x {np.array} -- The required 'x' variables to predict the result
        
        Returns:
            np.array -- Prediction of the data, if multiple data given as input an array is return as output
        """
        x = np.array(x)
        if len(x.shape) == 1 :
            temp = []
            temp.append(x)
            x = temp[:]
            del(temp)
        x = np.array(x)
        x = x.reshape(len(x),-1)
        y = self.model.predict(x)
        return y