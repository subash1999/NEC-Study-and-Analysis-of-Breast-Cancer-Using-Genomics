#used for the 100 genes classification
import sys, os
path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

import pandas as pd 
import numpy as npclear

from Classification_Models.decision_tree import GenomicsDTC, GenomicsDTR
from Classification_Models.k_nearest_neighbour import GenomicsKNN
from Classification_Models.naives_bayes import GenomicsBNB,GenomicsCNB,GenomicsGNB,GenomicsMNB 
from Classification_Models.support_vector_machine import GenomicsSVC
from Top_Ranking_Models.chi_square import GenomicsChiSquare
from record_performance import RecordPeformance

for x in range(0,10):
    print("Count : ",x)

    record = RecordPeformance("test")

    chi2 = GenomicsChiSquare()

    svc  = GenomicsSVC()
    knn = GenomicsKNN()
    bnb = GenomicsBNB()
    cnb = GenomicsCNB()
    gnb = GenomicsGNB()
    mnb = GenomicsMNB()
    dtc = GenomicsDTC()
    dtr = GenomicsDTR()

    

    record.top(chi2.method_name,100)
    record.top_start()
    chi2.makeTopGenesDF()
    df_top = chi2.getChi2TopGenesDF()
    record.top_end()

    record.tot(svc.clf_name,100,chi2.method_name)
    record.clf(svc.clf_name,100)
    record.tot_start()
    record.clf_start()
    svc.setDF(df_top)
    svc.trainModel()
    record.clf_end(svc.acc_train,svc.acc_test)
    record.tot_end(svc.acc_train,svc.acc_test)

    record.tot(knn.clf_name,100,chi2.method_name)
    record.clf(knn.clf_name,100)
    record.tot_start()
    record.clf_start()
    knn.setDF(df_top)
    knn.trainModel()
    record.clf_end(knn.acc_train,knn.acc_test)
    record.tot_end(knn.acc_train,knn.acc_test)

    record.tot(bnb.clf_name,100,chi2.method_name)
    record.clf(bnb.clf_name,100)
    record.tot_start()
    record.clf_start()
    bnb.setDF(df_top)
    bnb.trainModel()
    record.clf_end(bnb.acc_train,bnb.acc_test)
    record.tot_end(bnb.acc_train,bnb.acc_test)

    record.tot(cnb.clf_name,100,chi2.method_name)
    record.clf(cnb.clf_name,100)
    record.tot_start()
    record.clf_start()
    cnb.setDF(df_top)
    cnb.trainModel()
    record.clf_end(cnb.acc_train,cnb.acc_test)
    record.tot_end(cnb.acc_train,cnb.acc_test)

    record.tot(gnb.clf_name,100,chi2.method_name)
    record.clf(gnb.clf_name,100)
    record.tot_start()
    record.clf_start()
    gnb.setDF(df_top)
    gnb.trainModel()
    record.clf_end(gnb.acc_train,gnb.acc_test)
    record.tot_end(gnb.acc_train,gnb.acc_test)

    record.tot(mnb.clf_name,100,chi2.method_name)
    record.clf(mnb.clf_name,100)
    record.tot_start()
    record.clf_start()
    mnb.setDF(df_top)
    mnb.trainModel()
    record.clf_end(mnb.acc_train,mnb.acc_test)
    record.tot_end(mnb.acc_train,mnb.acc_test)

    record.tot(dtc.clf_name,100,chi2.method_name)
    record.clf(dtc.clf_name,100)
    record.tot_start()
    record.clf_start()
    dtc.setDF(df_top)
    dtc.trainModel()
    record.clf_end(dtc.acc_train,dtc.acc_test)
    record.tot_end(dtc.acc_train,dtc.acc_test)

    record.tot(dtr.clf_name,100,chi2.method_name)
    record.clf(dtr.clf_name,100)
    record.tot_start()
    record.clf_start()
    dtr.setDF(df_top)
    dtr.trainModel()
    record.clf_end(dtr.acc_train,dtr.acc_test)
    record.tot_end(dtr.acc_train,dtr.acc_test)

    del(record)

    del(chi2)

    del(svc)
    del(knn)
    del(bnb)
    del(cnb)
    del(gnb)
    del(mnb)
    del(dtc)
    del(dtr)