import pandas as pd 
import numpy as np
import os
from sklearn import preprocessing, svm
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

    def trainModel(self,test_size_input: int =0.2 ,number_of_jobs: int=-1) -> LinearRegression :
        """This method trains the model with linear regression, before it call makeDF()
        
        Keyword Arguments:
            test_size_input {float} -- Fraction of data used for testing (default: {0.2:float})
            number_of_jobs {int} --  number of jobs running parallely,
                                    -1 means maximum possible parallel jobs(default: {-1:int})
        
        Returns:
            sklearn.linear_model -- LinearRegression Model of the given data
        """
        x_train,x_test,y_train,y_test = self.splitDataToTrainTest(test_size_input) 
        clf = self.classification(x_train,y_train,number_of_jobs)
        y_test_predict = self.predictData(clf,x_test)
        accuracy_dict = self.accuracyOfModel(clf,x_train,y_train,y_test,y_test_predict)
        print("Accuracy Train : ",accuracy_dict['acc_train'])
        print("Accuracy Test : ",accuracy_dict['acc_test'])

        self.model = clf

        return clf

    

    def classification(self,x_train : np.array ,y_train : np.array ,number_of_jobs: int = -1 ) -> LinearRegression :        
        """This returns the Linear Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            LinearRegression -- It is a linear regression model
        """
        clf = LinearRegression(n_jobs = number_of_jobs)
        clf.fit(x_train,y_train)
        return clf

    def predictData(self, clf : LinearRegression, x : np.array) -> np.array :
        """Changes to the prediction of the linear model to either 0 or 1
        
        Arguments:
            clf {LinearRegression} -- Classifer model of LinearRegression
            x {np.array} -- The value of 'x' for which 'y' is to be predicted
        
        Returns:
            np.array -- The value predicted for 'x' input
        """        
        x = np.array(x)
        if len(x.shape) == 1 :
            temp = []
            temp.append(x)
            x = temp[:]
            del(temp)
        x = np.array(x)
        x = x.reshape(len(x),-1)
        y_out = clf.predict(x)
        y_test_predict = []
        for y in y_out :
            if y>= 0.5 :
                y_test_predict.append(1)
            else:
                y_test_predict.append(0)
            # y_test_predict.append(round(y))
        return np.array(y_test_predict)        
    
    def accuracyOfModel(self,clf : LinearRegression ,x_train : np.array ,y_train : np.array,y_test : np.array ,y_test_predict : np.array) -> dict:
        """Calculates accuracy of model
        
        Arguments:
            clf {LogicalRegression} -- Classifer model
            x_train {np.array} -- 'x' for train data
            y_train {np.array} -- 'y' for the data to train 
            y_test {np.array} -- 'y' of the test data
            y_test_predict {np.array} -- 'y' of the predicted test data
        
        Returns:
            dict -- The dictionary of the accuracy of train and test
        """
        self.acc_train = clf.score(x_train,y_train)
        self.acc_test = self.calculateAccuracy(y_test,y_test_predict)
        
        return {'acc_train': self.acc_train,'acc_test' : self.acc_test}

    def calculateAccuracy(self,y_exact : np.array ,y_predicted : np.array) -> float:
        """Calculate accuracy of the data
        
        Arguments:
            y_exact {np.array} -- 'y' of the exact data
            y_predicted {np.array} -- 'y' of the predicted data
        
        Returns:
            float -- accuracy score was made
        """     
        count = 0
        match = 0
        for k,y in enumerate(y_exact) :
            count += 1
            if y_predicted[k] == y:
                match += 1
        return match/count         

    def predict(self,x : np.array ) -> np.array :
        """This method predicts the value given the input data, use trainModel() before this method
        
        Arguments:
            x {np.array} -- The required 'x' variables to predict the result
        
        Returns:
            np.array -- Prediction of the data, if multiple data given as input an array is return as output
        """
        y = self.predictData(self.model , x )
        return y
