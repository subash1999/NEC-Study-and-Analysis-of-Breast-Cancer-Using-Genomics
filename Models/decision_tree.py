import pandas as pd
import numpy as np
from sklearn import tree

from project3_parent import Project3Parent

class GenomicsDT(Project3Parent): 

    def __int__(self):
        super().__init__()

    def trainModel(self, test_size_input = 0.2 ):
        x_train,x_test,y_train,y_test = self.splitDataToTrainTest(test_size_input) 
        clf = self.classification(x_train,y_train)
        
        self.model = clf

        return clf

    def classification(self, x_train : np.array , y_train : np.array ) -> tree :
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        return clf

    def accuracy(self):
        acc_train = self.model.score(self.x_train, self.y_train)
        acc_test = self.model.score (self.x_test, self.y_test)

        return {'acc_train':acc_train, 'acc_test' : acc_test}

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