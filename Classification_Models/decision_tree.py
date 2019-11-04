import numpy as np
import pandas as pd
from sklearn import tree
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from project3_parent import Project3Parent


class GenomicsDT(Project3Parent,ABC): 
    """Abstract Class For the "Decision Tree Method" : inherits Project3Parent"""
    def __int__(self):
        super().__init__()
        self.clf_name = "Decision Tree"

    def trainModel(self, test_size_input = 0.2 ):
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
    def classification(self, x_train : np.array , y_train : np.array ) -> tree :
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        return clf

    def accuracyOfModel(self,clf : tree ,x_train : np.array ,y_train : np.array,x_test : np.array ,y_test : np.array) -> dict:
        self.acc_train = self.model.score(self.x_train, self.y_train)
        self.acc_test = self.model.score (self.x_test, self.y_test)

        return {'acc_train':self.acc_train, 'acc_test' : self.acc_test}

    def predict(self, x : np.array ) -> np.array :
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

    def predict_probability(self, x : np.array) -> np.array : 
        x = np.array(x)
        if len(x.shape) == 1 :
            temp = []
            temp.append(x)
            x = temp[:]
            del(temp)
        x = np.array(x)
        x = x.reshape(len(x),-1)
        y = self.model.predict_proba(x)
        return y  

    def plotDecisionTree(self):
        plot = None
        if 'model' in globals():
            plot = tree.plot_tree(self.model)
        else : 
            self.trainModel()
            plot = tree.plot_tree(self.model)
        tree.plot_tree(self.model) 

        return plot 


class GenomicsDTC(GenomicsDT):
    """Decision Tree Classifier, a class that extends GenomicsDT Class
    Implements Decision Tree Classifier For Regression
    
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Decision Tree Classifier"
        
    def classification(self, x_train : np.array , y_train : np.array ) -> tree :
            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)

            return clf

class GenomicsDTR(GenomicsDT):
    
    """Decision Tree Regression, a class that extends GenomicsDT Class
    Implements Decision Tree Regression For Classification
    
    """
    def __init__(self):
        super().__init__()
        self.clf_name = "Decision Tree Regression"
        
    def classification(self, x_train : np.array , y_train : np.array ) -> tree :
            clf = tree.DecisionTreeRegressor()
            clf.fit(x_train, y_train)

            return clf