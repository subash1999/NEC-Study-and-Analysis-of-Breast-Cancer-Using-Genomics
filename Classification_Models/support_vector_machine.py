import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from project3_parent import Project3Parent

class GenomicsSVC(Project3Parent):
    
    def __init__(self):
        super().__init__()
        self.clf_name = "Support Vector Machine Classification"
    def trainModel(self,test_size_input: int =0.2) -> svm.SVC() :
        """This method trains the model with SVC classification, before it call makeDF()
        
        Keyword Arguments:
            test_size_input {float} -- Fraction of data used for testing (default: {0.2:float})
            
        Returns:
            svm.SVC() -- KNN Model of the given data
        """
        x_train,x_test,y_train,y_test = self.splitDataToTrainTest(test_size_input) 
        clf = self.classification(x_train,y_train)
        self.model = clf
        
        accuracy_dict = self.accuracyOfModel(clf,x_train,y_train,x_test,y_test)
        self.prec_train = precision_score(self.y_train,self.predict(x_train))
        self.prec_test = precision_score(self.y_test,self.predict(x_test))

        print("----"*3,self.clf_name,"----"*3)
        print("Accuracy Train : ",accuracy_dict['acc_train'])
        print("Accuracy Test : ",accuracy_dict['acc_test'])
        print("Precision Train : ",self.prec_train)
        print("Precision Test : ",self.prec_test)        

        self.conf_test = confusion_matrix(self.y_test,self.predict(x_test))
        self.conf_train = confusion_matrix(self.y_train,self.predict(x_train))
        print("Confusion Train(tp,tn,fp,fn): ",self.conf_train)
        print("Confusion Test(tp,tn,fp,fn) : ",self.conf_test)
        
        
        return clf

    def classification(self,x_train : np.array ,y_train : np.array) -> svm.SVC() :        
        """This returns the Support Vector Method Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Keyword Arguments:
            number_of_jobs {int} -- number of jobs done parallely (default: {-1:int})
        
        Returns:
            svm.SVC() -- It is a Support Vector Machine Classifier model
        """
        clf = svm.SVC()
        clf.fit(x_train,y_train)
        return clf

    def accuracyOfModel(self,clf : svm.SVC() ,x_train : np.array ,y_train : np.array,x_test : np.array ,y_test : np.array) -> dict:
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
