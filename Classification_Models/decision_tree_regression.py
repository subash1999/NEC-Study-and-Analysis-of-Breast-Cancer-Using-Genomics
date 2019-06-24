import numpy as np
from sklearn import tree

from decision_tree_parent import GenomicsDT

class GenomicsDTR(GenomicsDT):

    def __init__(self):
        super().__init__()
        
    def classification(self, x_train : np.array , y_train : np.array ) -> tree :
            clf = tree.DecisionTreeRegressor()
            clf.fit(x_train, y_train)

            return clf