import datetime
import time

import numpy as np
import pandas as pd

from record_best_model import RecordBestModel
from project3_parent import Project3Parent
import sys
import math


class UpdateBestModel(Project3Parent):

    def __init__(self):
        self.rec_per = RecordBestModel()
        self.df_existing = self.rec_per.getDFFromCSVFile()
        self.min_test_acc = 0
        self.min_train_acc = 0
        self.max_test_acc = 100
        self.max_train_acc = 100
        self.acceptable_acc_gap = 1
        self.is_best_model = False

    def updateBestModel(self,clf_model,no_of_genes,top_ranking_model=None):
        self.df_existing = self.rec_per.getDFFromCSVFile()
        rows_to_drop = self.checkBestModel(clf_model,top_ranking_model)
        # print("rows_to_drop : : ",rows_to_drop)
        self.df_existing = self.df_existing.drop(rows_to_drop)
        self.rec_per.setDFToCSVFile(self.df_existing)
        if self.is_best_model :            
            self.rec_per.recordPerformance(clf_model,no_of_genes,top_ranking_model)

    def checkBestModel(self,clf_model,top_ranking_model = None):
        rows_to_drop = []
        top_ranking_model_name = "No Model"
        if (top_ranking_model != None):
            top_ranking_model_name = top_ranking_model.method_name
        # temp_df  = self.df_existing[ (self.df_existing['Clf_Model'] == clf_model.clf_name)]
        # print(temp_df)
        self.df = self.df_existing[:]
        rows_to_check = self.df[ (self.df['Clf_Model'] == clf_model.clf_name) & (self.df['Top_Ranking_Model'] == top_ranking_model_name)].index
        print("---------------------"*20+"\n"+"---------"*20+"\n"+"---------"*20)
        print(rows_to_check)
   
        if (clf_model.acc_test >= self.min_test_acc and 
            clf_model.acc_test <= self.max_test_acc and 
            clf_model.acc_train >= self.min_train_acc and  
            clf_model.acc_train <= self.max_train_acc ):        
            if (len(rows_to_check) != 0):                
                acc_gap = abs(clf_model.acc_test - clf_model.acc_train)
                if acc_gap <= self.acceptable_acc_gap :
                    for row_index in rows_to_check:
                        print("rows to check : ",rows_to_check)
                        print("row index : ",row_index)
                        print(self.df_existing)
                        print("---------------------"*20)
                        
                        print(self.df_existing.iloc[row_index])
                        row = self.df_existing.iloc[row_index]
                        # print(row)
                        if (row["Train_Acc"] > clf_model.acc_train or
                         row["Test_Acc"] > clf_model.acc_test):
                            rows_to_drop.append(row_index)
                            self.is_best_model = True
            else :
                self.is_best_model = True
        return rows_to_drop        
                
