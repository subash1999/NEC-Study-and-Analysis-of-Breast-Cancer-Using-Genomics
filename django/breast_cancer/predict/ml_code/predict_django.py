import pandas as pd
import numpy as np
import sys,os,gc
path = os.path.abspath(os.path.join('..', '..',))
sys.path.append(path)
from project3_parent import Project3Parent
import time

class PredictDjango():
    def __init__(self):
        self.project3_parent = Project3Parent('../../Data/supervised_learning/output_dataset/gse_2034_processed_data.csv')
        self.best_model_csv_path = "../../best_trained_models/best_performance.csv"
        self.best_model_csv_df = pd.read_csv(self.best_model_csv_path)
        self.selected_model = None
        self.data_file_name = None 
        self.selected_df = None   
    def getBestModelCSVDF(self):
        return self.best_model_csv_df
    def getSelectedModel(self):
        return self.selected_model
    def selectModel(self,index):
        filename_without_extension = self.best_model_csv_df.iloc[index]["File_Name"].split('.pkl')[0]
        self.selected_model = self.project3_parent.getModelUsingJoblib("../../best_trained_models/"+filename_without_extension)
        self.data_file_name = "../../best_trained_models/csv/"+str(filename_without_extension)+".csv"
        self.selected_df = pd.read_csv(self.data_file_name)
        return self.selected_model
    def getCSVOfIndex(self,index):
        filename_without_extension = self.best_model_csv_df.iloc[index]["File_Name"].split('.pkl')[0]
        data_file_name = "../../best_trained_models/csv/"+str(filename_without_extension)+".csv"
        return pd.read_csv(data_file_name)   
    def predictResult(self,test_index):
        (x,y) = self.project3_parent.getXYOfData(self.selected_df)
        to_predict = x[test_index]
        actual_result = y[test_index]
        print('Selected Model :',self.selectModel)
        predict_result = self.selected_model.predict(to_predict)
        return {
            'actual' : actual_result,
            'predicted' : predict_result,
        }

        
    def getSelectedDF(self):
        return self.selected_df
