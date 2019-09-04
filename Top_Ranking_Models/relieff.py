from ReliefF import ReliefF
import numpy as np
from sklearn import datasets
import pandas as pd
from project3_parent import Project3Parent 

class GenomicsReliefF(Project3Parent):
    def __init__(self):
        super().__init__()
        self.reliefF_df = None
        self.method_name = "ReliefF Method"

    def selectTopGenes(self,x : np.array,y : np.array,n_neighbors = 20, top_k_genes = 100) -> np.array:
        """It select the top genes/features using ReliefF algorithm from REliefF package
        
        Arguments:
            x {np.array} -- The features
            y {np.array} -- the result i.e. dependent variable
        
        Keyword Arguments:
            n_neighbours {int} -- The numbers of neighbours to consider while applying algorithm (default: {20})
            top_k_genes {int} -- Number of top_genes we need to select (default: {100})
        
        Returns:
            np.array -- returns the top features selected among features in x variable given
        """
        feature_selector = ReliefF(n_neighbors=20, n_features_to_keep=top_k_genes)
        x_new = feature_selector.fit_transform(x,y)
        # for x in feature_selector.feature_scores:
        #     print(x,end=',')
        print(np.amax(feature_selector.feature_scores))
        return x_new

    def makeTopGenesDF(self,top_k_genes : int = 100):
        """Select the top genes and make df from it
        
        Keyword Arguments:
            top_k_genes {int} -- number of top features we need (default: {100})
        """
        x_new = self.selectTopGenes(self.x,self.y,top_k_genes)
        self.reliefF_df =  self.returnDFfromNP(x_new,self.y)
    
    def getReliefFTopGenesDF(self)->pd.DataFrame:
        """Returns the df of chi2 top ranked genes
        *** Use makeTopGenesDF() before this otherwise None is returned ***
        
        Returns:
            pd.DataFrame -- dataframe of top ranked genes by chi square method
        """
        if type(self.reliefF_df) != None:
            return self.reliefF_df
        else:
            print("Call makeTopGeneDF() before this function, Now only none is returned")
    