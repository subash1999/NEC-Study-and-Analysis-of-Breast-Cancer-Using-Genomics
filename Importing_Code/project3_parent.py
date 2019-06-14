import pandas as pd 
import numpy as np
import os

class Project3Parent():
    """This is the parent class for the all machine learning of project3 
    This contains the default path of gene data, 
    functions to manipulate the current_path,path and DataFrame
    
    ATTRIBUTE
    ---------
    current_path = type:String, 
    csv_path = type:String,

    FUNCTION
    --------

    """

    def __init__(self):
        self.current_path = os.getcwd()
        self.csv_path = 'G:\Subash\BECOMP\Project 3\Project Coding Part\Project Code\Study-and-Analysis-of-Breast-Cancer-Using-Genomics\Data\supervised_learning\output_dataset\gse_2034_processed_data.csv'
        self.df = None
        # making dataframe
        self.makeDF()
    
    # getting the current path of command line
    def getCurrentPath(self) -> str: 
        """This returns the current path of the terminal
        
        Returns:
            String -- returns the current path of command line
        """
        return self.current_path

    # make the df from the csv_relative_path and return it
    def makeDF(self) -> pd.DataFrame :
        """This makes the data frame from the given csv_relative_path 
        
        Returns:
            pd.DataFrame -- dataframe of the genedata if file not found in 
            location this method will make df none ,pandas dataframe
        """
        try :
            self.df = pd.read_csv(self.csv_path,sep=',')
        except Exception as E: 
            print("------------"*10)
            print("\t"*10,"***** FILE NOT FOUND *****")
            print("------------"*10)
            print("File Not Found... \nPlease provide a proper path using setCsvPath(csv_path:str) method")
            print("\t"*5,"-----"*7," Exception", "-----"*7,"\n",E)
            print("------------"*10)
            print("\t"*10,"***** FILE NOT FOUND END *****")
            print("------------"*10)
            self.df = None
        return self.df 
    
    # get the DataFrame of the geneData if present 
    def getDF(self) -> pd.DataFrame : 
        """This method returns dataframe of the data
        
        Returns:
            pd.DataFrame -- dataframe of the gene data if present and 
            if not present this returns none
        """
        return self.df

    # get csv relative path
    def getCsvPath(self) -> str:    
        """This function path to the csv file
        
        Returns:
            String -- returns the dataframe of the csv data
        """ 
        return self.csv_path
    
    # getting the current path of command line
    def setCurrentPath(self,current_path: str): 
        """Set the Current path of the terminal
        
        Prams:
            String -- set the current path of command line
        """
        self.current_path = current_path

    # make the df from the csv_relative_path
    def setDF(self,df: pd.DataFrame ):
        """Set the dataframe i.e 
        Prams: 
            DataFrame -- data frame of gene data
        """
        self.df = df        

    # get csv path
    def setCsvPath(self,csv_path : str) -> str:    
        """This function sets path to the csv file
        Prams: 
            String -- set the current path of command line
        """ 
        self.csv_path = csv_path
