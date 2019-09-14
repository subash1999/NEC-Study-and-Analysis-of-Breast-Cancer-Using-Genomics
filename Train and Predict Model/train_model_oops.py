import gc
import os
import sys

import numpy as np
import pandas as pd

from Classification_Models.decision_tree import GenomicsDTC, GenomicsDTR
from Classification_Models.k_nearest_neighbour import GenomicsKNN
from Classification_Models.naives_bayes import (GenomicsBNB, GenomicsCNB,
                                                GenomicsGNB, GenomicsMNB)
from Classification_Models.support_vector_machine import GenomicsSVC
from record_performance import RecordPeformance
from Top_Ranking_Models.chi_square import GenomicsChiSquare
from update_best_model import UpdateBestModel

path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

class TrainModel():
    
    def __init__(self,top_ranking_method,number_of_genes = None):
        self.initializeVariables(top_ranking_method,number_of_genes)

    def initializeVariables(self,top_ranking_method,no_of_genes_to_filter):
        self.update_best_model_obj = UpdateBestModel()
        self.record = RecordPeformance("test")
        self.no_of_runs= 1
        self.top_ranking_method = top_ranking_method
        self.no_of_genes_to_filter = no_of_genes_to_filter
        
        self.svcs = []
        self.knns = []
        self.bnbs = []
        self.cnbs = []
        self.gnbs = []
        self.mnbs = []
        self.dtcs = []

        self.svc = GenomicsSVC()
        self.knn = GenomicsKNN()
        self.bnb = GenomicsBNB()
        self.cnb = GenomicsCNB()
        self.gnb = GenomicsGNB()
        self.mnb = GenomicsMNB()
        self.dtc = GenomicsDTC()

    def getTopDf(self):
        self.record.top(top_ranking_method.method_name,no_of_genes)
        self.record.top_start()
        top_ranking_method.makeTopGenesDF()
        df_top = top_ranking_method.getChi2TopGenesDF()
        self.record.top_end()
        return df_top

    def trainAndSaveModel(self):
        for x in range(0,self.no_of_runs):
            print("Count : ",x)
            record = RecordPeformance("test")

            record.top(self.top_gene_method.method_name,no_of_genes)
            record.top_start()
            self.top_gene_method.makeTopGenesDF(no_of_genes)
            df_top = self.top_gene_method.getChi2TopGenesDF()
            record.top_end()

            record.tot(svc.clf_name,no_of_genes,chi2.method_name)
            record.clf(svc.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            svc.setDF(df_top)
            svc.trainModel()
            
            record.clf_end(svc.acc_train,svc.acc_test)
            record.tot_end(svc.acc_train,svc.acc_test)

            record.tot(knn.clf_name,no_of_genes,chi2.method_name)
            record.clf(knn.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            knn.setDF(df_top)
            knn.trainModel()
            record.clf_end(knn.acc_train,knn.acc_test)
            record.tot_end(knn.acc_train,knn.acc_test)

            record.tot(bnb.clf_name,no_of_genes,chi2.method_name)
            record.clf(bnb.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            bnb.setDF(df_top)
            bnb.trainModel()
            record.clf_end(bnb.acc_train,bnb.acc_test)
            record.tot_end(bnb.acc_train,bnb.acc_test)

            record.tot(cnb.clf_name,no_of_genes,chi2.method_name)
            record.clf(cnb.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            cnb.setDF(df_top)
            cnb.trainModel()
            record.clf_end(cnb.acc_train,cnb.acc_test)
            record.tot_end(cnb.acc_train,cnb.acc_test)

            record.tot(gnb.clf_name,no_of_genes,chi2.method_name)
            record.clf(gnb.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            gnb.setDF(df_top)
            gnb.trainModel()
            record.clf_end(gnb.acc_train,gnb.acc_test)
            record.tot_end(gnb.acc_train,gnb.acc_test)

            record.tot(mnb.clf_name,no_of_genes,chi2.method_name)
            record.clf(mnb.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            mnb.setDF(df_top)
            mnb.trainModel()
            record.clf_end(mnb.acc_train,mnb.acc_test)
            record.tot_end(mnb.acc_train,mnb.acc_test)

            record.tot(dtc.clf_name,no_of_genes,chi2.method_name)
            record.clf(dtc.clf_name,no_of_genes)
            record.tot_start()
            record.clf_start()
            dtc.setDF(df_top)
            dtc.trainModel()
            record.clf_end(dtc.acc_train,dtc.acc_test)
            record.tot_end(dtc.acc_train,dtc.acc_test)

            

            del(record)

            svcs.append(svc)
            del(svc)
            knns.append(knn)
            del(knn)
            bnbs.append(bnb)
            del(bnb)
            cnbs.append(cnb)
            del(cnb)
            gnbs.append(gnb)
            del(gnb)
            mnbs.append(mnb)
            del(mnb)
            dtcs.append(dtc)
            del(dtc)
        
        
        svc_max = svcs[0]
        knn_max = knns[0]
        bnb_max = bnbs[0]
        cnb_max = cnbs[0]
        gnb_max = gnbs[0]
        mnb_max = mnbs[0]
        dtc_max = dtcs[0]
        # for i in range(0,no_of_runs):
        #     if svcs[i].acc_test > svc_max.acc_test:
        #         svc_max = svcs[i].acc_test
        #     if knns[i].acc_test > knn_max.acc_test:
        #         knn_max = knns[i].acc_test
        #     if bnbs[i].acc_test > bnb_max.acc_test:
        #         bnb_max = bnbs[i].acc_test
        #     if cnbs[i].acc_test > cnb_max.acc_test:
        #         cnb_max = cnbs[i].acc_test
        #     if gnbs[i].acc_test > gnb_max.acc_test:
        #         gnb_max = gnbs[i].acc_test
        #     if mnbs[i].acc_test > mnb_max.acc_test:
        #         mnb_max = mnbs[i].acc_test
        #     if dtcs[i].acc_test > dtc_max.acc_test:
        #         dtc_max = dtcs[i].acc_test
    
        svc_max.saveModelUsingJoblib(svc_max.model,"trained_models/svc_chi_square_"+str(no_of_genes))
        u.updateBestModel(svc_max,no_of_genes,chi2)
        knn_max.saveModelUsingJoblib(knn_max.model,"trained_models/knn_chi_square_"+str(no_of_genes))
        u.updateBestModel(knn_max,no_of_genes,chi2)
        bnb_max.saveModelUsingJoblib(bnb_max.model,"trained_models/bnb_chi_square_"+str(no_of_genes))
        u.updateBestModel(bnb_max,no_of_genes,chi2)
        cnb_max.saveModelUsingJoblib(cnb_max.model,"trained_models/cnb_chi_square_"+str(no_of_genes))
        u.updateBestModel(cnb_max,no_of_genes,chi2)
        gnb_max.saveModelUsingJoblib(gnb_max.model,"trained_models/gnb_chi_square_"+str(no_of_genes))
        u.updateBestModel(gnb_max,no_of_genes,chi2)
        mnb_max.saveModelUsingJoblib(mnb_max.model,"trained_models/mnb_chi_square_"+str(no_of_genes))
        u.updateBestModel(mnb_max,no_of_genes,chi2)
        dtc_max.saveModelUsingJoblib(dtc_max.model,"trained_models/dtc_chi_square_"+str(no_of_genes))
        u.updateBestModel(dtc_max,no_of_genes,chi2)