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

path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

u = UpdateBestModel()
no_of_runs=1

def chi_square_models_save(no_of_genes):
    svcs = []
    knns = []
    bnbs = []
    cnbs = []
    gnbs = []
    mnbs = []
    dtcs = []
    
    chi2 = GenomicsChiSquare()

    for x in range(0,no_of_runs):
        print("Count : ",x)

        record = RecordPeformance("test")


        svc  = GenomicsSVC()
        knn = GenomicsKNN()
        bnb = GenomicsBNB()
        cnb = GenomicsCNB()
        gnb = GenomicsGNB()
        mnb = GenomicsMNB()
        dtc = GenomicsDTC()
        
        

        record.top(chi2.method_name,no_of_genes)
        record.top_start()
        chi2.makeTopGenesDF(no_of_genes)
        df_top = chi2.getChi2TopGenesDF()
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
    
def all_genes_models():
    svcs = []
    knns = []
    bnbs = []
    cnbs = []
    gnbs = []
    mnbs = []
    dtcs = []
    for x in range(0,no_of_runs):
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
    
    svc_max.saveModelUsingJoblib(svc_max.model,"trained_models/svc_all_genes_22385")
    u.updateBestModel(svc_max,222385)
    knn_max.saveModelUsingJoblib(knn_max.model,"trained_models/knn_all_genes_22385")
    u.updateBestModel(knn_max,222385)
    bnb_max.saveModelUsingJoblib(bnb_max.model,"trained_models/bnb_all_genes_22385")
    u.updateBestModel(bnb_max,222385)
    cnb_max.saveModelUsingJoblib(cnb_max.model,"trained_models/cnb_all_genes_22385")
    u.updateBestModel(cnb_max,222385)
    gnb_max.saveModelUsingJoblib(gnb_max.model,"trained_models/gnb_all_genes_22385")
    u.updateBestModel(gnb_max,222385)
    mnb_max.saveModelUsingJoblib(mnb_max.model,"trained_models/mnb_all_genes_22385")
    u.updateBestModel(mnb_max,222385)
    dtc_max.saveModelUsingJoblib(dtc_max.model,"trained_models/dtc_all_genes_22385")
    u.updateBestModel(dtc_max,222385)

# gc.collect()
# all_genes_models()
gc.collect()
for x in range(100,22385,100):
    chi_square_models_save(x)
    gc.collect()
