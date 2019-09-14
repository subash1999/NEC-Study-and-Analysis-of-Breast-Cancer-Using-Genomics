import pandas as pd
import numpy as np
import sys,os,gc
from project3_parent import Project3Parent
import time

class Predict():
    def __init__(self):
        self.project3_parent = Project3Parent()
        pd.set_option('display.max_rows', 30)
        pd.set_option('display.max_columns', 4)
        pd.set_option('display.width', 1000)
        self.best_model_csv_path = "best_trained_models/best_performance.csv"
        self.best_model_csv_df = pd.read_csv(self.best_model_csv_path)
        self.selected_model = None 
        self.selected_df = None   
        self.operations()

    def operations(self):
        self.showOptions()

        model_index = self.getValidInput("Choose A Model Index : ",list(self.best_model_csv_df.index.values))

        filename_without_extension = self.best_model_csv_df.iloc[model_index]["File_Name"].split('.pkl')[0]
        self.model = self.project3_parent.getModelUsingJoblib("best_trained_models/"+filename_without_extension)
        self.data_file_name = "best_trained_models/csv/"+str(filename_without_extension)+".csv"
        
        self.data_file_df = pd.read_csv(self.data_file_name)

        self.showFileData(self.data_file_df)

        test_index = self.getValidInput(" Choose a row to test model : ",list(self.data_file_df.index.values))
        (x,y) = self.project3_parent.getXYOfData(self.data_file_df)
        to_predict = x[test_index]
        actual_result = y[test_index]

        predict_result = self.model.predict(to_predict)

        print("Predicted Result : ",predict_result)
        print("Actual Result : ",actual_result)

        contd = input("\n\n"+"****DO YOU WANT TO CONTINUE ??? Y/N : ")
        if (contd.lower() == "y"):
            self.operations()
        else:
            thank_you = "\t"*7+"*"*50
            thank_you = thank_you + "\n"+"\t"*7+"***** THANK YOU FOR USING OUR PROGRAM *****"
            thank_you = thank_you + "\n"+"\t"*7+"*"*50
            print(thank_you)
        input()
        
    def showOptions(self):
        print("---///"*30)
        print(("\t"*7+"**********STUDY AND ANALYSIS OF BREAST CANCER WITH GENOMICS**********\n")*3)
        print("---///"*30)
        print(("\t"*7+"----BEST MODEL DATAFRAME----\n")*3)
        print(self.best_model_csv_df)
        print("-----"*50)
    
    def showFileData(self,data_file_df):
        print(("\t"*5+"----DATA TO TEST FOR SELECTED MODEL----\n")*3)
        print(data_file_df)

    def getValidInput(self,msg,valid_input_list):
        print("Available Options : \n",valid_input_list)
        valid_input_list = list(valid_input_list)
        input_list = []
        for x in valid_input_list:
            input_list.append(str(x))
        val = input(msg)        
        if (val in input_list):
            return int(val)
        else:
            print("Entered : ",val)
            print("Invalid Input Please Enter a valid input")
            self.getValidInput(msg,valid_input_list)


    def getInput(self):
        pass

# p = Predict()