import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors

from project3_parent import Project3Parent

class GenomicsKNN(Project3Parent):
    
    def __init__(self):
        super().__init__()

    def trainModel(self,test_size_input: int =0.01) -> neighbors.KNeighborsClassifier() :
        """This method trains the model with KNN classification, before it call makeDF()
        
        Keyword Arguments:
            test_size_input {float} -- Fraction of data used for testing (default: {0.2:float})
            
        Returns:
            sklearn.linear_model -- KNN Model of the given data
        """
        x_train,x_test,y_train,y_test = self.splitDataToTrainTest(test_size_input) 
        clf = self.classification(x_train,y_train)
        accuracy_dict = self.accuracyOfModel(clf,x_train,y_train,x_test,y_test)
        print("Accuracy Train : ",accuracy_dict['acc_train'])
        print("Accuracy Test : ",accuracy_dict['acc_test'])

    def classification(self,x_train : np.array ,y_train : np.array) -> neighbors.KNeighborsClassifier() :        
        """This returns the KNeighbours Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            neighbors.KNeighborsClassifier -- It is a KNeighborsClassifier model
        """
        clf = neighbors.KNeighborsClassifier()
        clf.fit(x_train,y_train)
        return clf

    def accuracyOfModel(self,clf : neighbors ,x_train : np.array ,y_train : np.array,x_test : np.array ,y_test : np.array) -> dict:
        """Calculates accuracy of model
        
        Arguments:
            clf {neighbours} -- KNN Classifer model
            x_train {np.array} -- 'x' for train data
            y_train {np.array} -- 'y' for the data to train 
            y_test {np.array} -- 'y' of the test data
            y_test_predict {np.array} -- 'y' of the predicted test data
        
        Returns:
            dict -- The dictionary of the accuracy of train and test
        """
        self.acc_train = clf.score(x_train,y_train)
        self.acc_test = clf.score(x_test,y_test)
        
        return {'acc_train': self.acc_train,'acc_test' : self.acc_test}
