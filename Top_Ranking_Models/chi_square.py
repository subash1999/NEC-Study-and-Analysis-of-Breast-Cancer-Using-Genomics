import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from project3_parent import Project3Parent

class GenomicsChiSquare(Project3Parent):

    def __init__(self):
        super().__init__()
        self.chi2_df = None
        self.method_name = "Chi Square (X²) Method"

    def selectTopGenes(self,x : np.array,y : np.array, top_k_genes = 100) -> np.array:
        """It select the top genes/features
        
        Arguments:
            x {np.array} -- The features
            y {np.array} -- the result i.e. dependent variable
        
        Keyword Arguments:
            top_k_genes {int} -- Number of top_genes we need (default: {100})
        
        Returns:
            np.array -- returns the top features selected among features in x variable given
        """
        x_new = SelectKBest(chi2, k=top_k_genes).fit_transform(x,y)
        return x_new

    def makeTopGenesDF(self,top_k_genes : int = 100):
        """Select the top genes and make df from it
        
        Keyword Arguments:
            top_k_genes {int} -- number of top features we need (default: {100})
        """
        x_new = self.selectTopGenes(self.x,self.y,top_k_genes)
        self.chi2_df =  self.returnDFfromNP(x_new,self.y)
    
    def getChi2TopGenesDF(self)->pd.DataFrame:
        """Returns the df of chi2 top ranked genes
        *** Use makeTopGenesDF() before this otherwise None is returned ***
        
        Returns:
            pd.DataFrame -- dataframe of top ranked genes by chi square method
        """
        if type(self.chi2_df) != None:
            return self.chi2_df
        else:
            print("Call makeTopGeneDF() before this function, Now only none is returned")


        