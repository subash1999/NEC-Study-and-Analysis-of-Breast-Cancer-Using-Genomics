import datetime
import time

import numpy as np
import pandas as pd
import sys

from project3_parent import Project3Parent

class RecordBestModel(Project3Parent):

    def __init__(self,run_type = None):
        """Initialize the path, save_file_name, df_columns and finally get df from file if file exists 
        if not exists create a new df with the df_columns in that df.
    
        Raises:
            ValueError: value error raised from getDFFromCSVFile() when the colums of csv doesnot
                match with the required columns  for recording best performance
        """
        self.path = "best_trained_models/"
        self.csv_file_name = "best_performance.csv" 
        self.df_columns = ['Clf_Model','Top_Ranking_Model','No_Of_Genes','Train_Acc','Test_Acc','File_Name','Time']
        self.df = self.getDFFromCSVFile()    
    
    def recordPerformance(self,clf_model,no_of_genes:int,top_ranking_model = None) -> str:
        """Records the performance for the best performance given by a model and save it in a csv file
            whose name and path is set in the path mentioned in  __init__()  
        Arguments : 
                clf_model : an object of classification model from classification_model folder
                top_ranking_model : an object of top ranking model from top_ranking_folder
                no_of_genes {init} : number of genes filtered using top_ranking_model for better
                                        feature selection
        """
        top_ranking_model_name = "No Model"
        if (top_ranking_model != None):
            top_ranking_model_name = top_ranking_model.method_name

        # rows to delete if the clf_Model and the top_ranking_model matches to the existing one
        rows_to_drop = self.df[ (self.df['Clf_Model'] == clf_model.clf_name) & (self.df['Top_Ranking_Model'] == top_ranking_model_name)].index
        
        # Delete these row indexes from dataFrame
        self.df = self.df.drop(rows_to_drop)

        listOfSeries = [pd.Series([
            clf_model.clf_name,
            top_ranking_model_name,
            no_of_genes,
            clf_model.acc_train,
            clf_model.acc_test,
            clf_model.clf_name+"_"+top_ranking_model_name+".pkl",
            str(datetime.datetime.now()),
        ], index=self.df.columns ) ]
        self.df = self.df.append(listOfSeries , ignore_index=True)
        
        self.saveModelUsingJoblib(clf_model,"best_trained_models/"+clf_model.clf_name+"_"+top_ranking_model_name)
        clf_model.getDF().to_csv(self.path+"csv/"+clf_model.clf_name+"_"+top_ranking_model_name+".csv", index=False)
        self.df.to_csv(self.path+self.csv_file_name, index=False)        
    

    def getDFFromCSVFile(self):
        """Checks the csv file existance for recording the best model
        If file not exists create new df
        If file exists but columns doesnot match with required columns create new df
        If file exists and contains the required colums use it as df
        
        Raises:
            ValueError: alue error raised when the colums of csv doesnot
            match with the required columns  for recording best performance
        """
        df = pd.DataFrame()

        try:
            df = pd.read_csv(self.path+self.csv_file_name,sep=",")
            if (df.columns.tolist() != self.df_columns):
                raise ValueError("Columns of CSV doesnot match with required columns")
        except FileNotFoundError:
            print("File Not Found Error")
            df = pd.DataFrame(columns=self.df_columns)
        # except ValueError as e:
        #     print(e)
        #     sys.exit()
        #     df = pd.DataFrame(columns=self.df_columns)
        # except Exception as e:
        #     print("Some Uncalculated Error Occured")
        #     print(e)
        #     sys.exit()

        return df

    def setDFToCSVFile(self,df):
        df.to_csv(self.path+self.csv_file_name, sep=',', index=False)
