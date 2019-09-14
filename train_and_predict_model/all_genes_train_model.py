#used for the 100 genes classification
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
from project3_parent import Project3Parent

path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

class AllGenesTrainModel():

    def __init__(self):
        project3 = Project3Parent()
        self.no_of_genes = len(project3.getDF().columns)-2
        self.no_of_runs = 1

    def trainAndSaveModels(self):
        u = UpdateBestModel()

        svcs = []
        knns = []
        bnbs = []
        cnbs = []
        gnbs = []
        mnbs = []
        dtcs = []

        for x in range(0,self.no_of_runs):
            print("Count : ",x)

            record = RecordPeformance("test")

        
            svc  = GenomicsSVC()
            knn = GenomicsKNN()
            bnb = GenomicsBNB()
            cnb = GenomicsCNB()
            gnb = GenomicsGNB()
            mnb = GenomicsMNB()
            dtc = GenomicsDTC()
            
            
            record.clf(svc.clf_name,svc.getDF().shape[1])
            record.clf_start()
            svc.trainModel()
            record.clf_end(svc.acc_train,svc.acc_test)
            svcs.append(svc)
            del(svc)
            
            record.clf(knn.clf_name,knn.getDF().shape[1])
            record.clf_start()
            knn.trainModel()
            record.clf_end(knn.acc_train,knn.acc_test)
            knns.append(knn)
            del(knn)

            record.clf(bnb.clf_name,bnb.getDF().shape[1])
            record.clf_start()
            bnb.trainModel()
            record.clf_end(bnb.acc_train,bnb.acc_test)
            bnbs.append(bnb)
            del(bnb)

            record.clf(cnb.clf_name,cnb.getDF().shape[1])
            record.clf_start()
            cnb.trainModel()
            record.clf_end(cnb.acc_train,cnb.acc_test)
            cnbs.append(cnb)
            del(cnb)

            record.clf(gnb.clf_name,gnb.getDF().shape[1])
            record.clf_start()
            gnb.trainModel()
            record.clf_end(gnb.acc_train,gnb.acc_test)
            gnbs.append(gnb)
            del(gnb)

            record.clf(mnb.clf_name,mnb.getDF().shape[1])
            record.clf_start()
            mnb.trainModel()
            record.clf_end(mnb.acc_train,mnb.acc_test)
            mnbs.append(mnb)
            del(mnb)

            record.clf(dtc.clf_name,dtc.getDF().shape[1])
            record.clf_start()
            dtc.trainModel()
            record.clf_end(dtc.acc_train,dtc.acc_test)
            dtcs.append(dtc)
            del(dtc)

            
            del(record) 
        

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
        
        svc_max.saveModelUsingJoblib(svc_max.model,"trained_models/svc_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(svc_max,self.no_of_genes)
        knn_max.saveModelUsingJoblib(knn_max.model,"trained_models/knn_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(knn_max,self.no_of_genes)
        bnb_max.saveModelUsingJoblib(bnb_max.model,"trained_models/bnb_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(bnb_max,self.no_of_genes)
        cnb_max.saveModelUsingJoblib(cnb_max.model,"trained_models/cnb_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(cnb_max,self.no_of_genes)
        gnb_max.saveModelUsingJoblib(gnb_max.model,"trained_models/gnb_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(gnb_max,self.no_of_genes)
        mnb_max.saveModelUsingJoblib(mnb_max.model,"trained_models/mnb_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(mnb_max,self.no_of_genes)
        dtc_max.saveModelUsingJoblib(dtc_max.model,"trained_models/dtc_all_genes_"+str(self.no_of_genes))
        u.updateBestModel(dtc_max,self.no_of_genes)


