import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,ComplementNB
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from project3_parent import Project3Parent

class GenomicsNB(Project3Parent,ABC) :
    """ Abstract Class for the Naives Bayes Method of classification : inherits Project3Parent """
    def __init__(self):
        super().__init__()
        self.clf_name = "Naives Bayes"

    def trainModel(self,test_size_input: int =0.01) -> GaussianNB() :
        """This method trains the model with KNN classification, before it call makeDF()
        
        Keyword Arguments:
            test_size_input {float} -- Fraction of data used for testing (default: {0.2:float})
            
        Returns:
            GaussianNB -- KNN Model of the given data
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
    @abstractmethod
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

    def accuracyOfModel(self,clf : GaussianNB ,x_train : np.array ,y_train : np.array,x_test : np.array ,y_test : np.array) -> dict:
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

class GenomicsMNB(GenomicsNB) :
    """
    Classifies using Multinomial Naives Bayes Classification : inherits GenomicsNB Class
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Multinomial Naives Bayes Classification"

    def classification(self,x_train : np.array ,y_train : np.array) -> MultinomialNB() :        
        """This returns the MultinomialNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Returns:
            MultinomialNB -- It is a KNeighborsClassifier model
        """
        clf = MultinomialNB()
        clf.fit(x_train,y_train)        
        return clf

class GenomicsBNB(GenomicsNB) :
    """
     Classifies using Bernoulli Naives Bayes Classification : inherits GenomicsNB Class
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Bernoulli Naives Bayes Classification"

    def classification(self,x_train : np.array ,y_train : np.array) -> BernoulliNB() :        
        """This returns the BernoulliNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Returns:
            BernoulliNB -- It is a KNeighborsClassifier model
        """
        clf = BernoulliNB()
        clf.fit(x_train,y_train)        
        return clf


class GenomicsCNB(GenomicsNB) :
    """
     Classifies using Complement Naives Bayes Classification : inherits GenomicsNB Class
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Complement Naives Bayes Classification"

    def classification(self,x_train : np.array ,y_train : np.array) -> ComplementNB() :        
        """This returns the ComplementNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Returns:
            ComplementNB -- It is a KNeighborsClassifier model
        """
        clf = ComplementNB()
        clf.fit(x_train,y_train)        
        return clf

class GenomicsGNB(GenomicsNB) :
    """
    Classifies using Gaussian Naives Bayes Classification : inherits GenomicsNB Class
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Gaussian Naives Bayes Classification"

    def classification(self,x_train : np.array ,y_train : np.array) -> GaussianNB() :        
        """This returns the GaussianNB Classification Model
        
        Arguments:
            x_train {np.array} -- The train data x
            y_train {np.array} -- The train data y
        
        Returns:
            GaussianNB -- It is a KNeighborsClassifier model
        """
        clf = GaussianNB()
        clf.fit(x_train,y_train)        
        return clf
