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

   
    svc  = GenomicsSVC()
    knn = GenomicsKNN()
    bnb = GenomicsBNB()
    cnb = GenomicsCNB()
    gnb = GenomicsGNB()
    mnb = GenomicsMNB()
    dtc = GenomicsDTC()
    dtr = GenomicsDTR()

      
    record.clf(svc.clf_name,svc.getDF().shape[1])
    record.clf_start()
    svc.trainModel()
    del(svc)
    
    record.clf(knn.clf_name,knn.getDF().shape[1])
    record.clf_start()
    knn.trainModel()
    record.clf_end(knn.acc_train,knn.acc_test)
    del(knn)

    record.clf(bnb.clf_name,bnb.getDF().shape[1])
    record.clf_start()
    bnb.trainModel()
    record.clf_end(bnb.acc_train,bnb.acc_test)
    del(bnb)

    record.clf(cnb.clf_name,cnb.getDF().shape[1])
    record.clf_start()
    cnb.trainModel()
    record.clf_end(cnb.acc_train,cnb.acc_test)
    del(cnb)

    record.clf(gnb.clf_name,gnb.getDF().shape[1])
    record.clf_start()
    gnb.trainModel()
    record.clf_end(gnb.acc_train,gnb.acc_test)
    del(gnb)

    record.clf(mnb.clf_name,mnb.getDF().shape[1])
    record.clf_start()
    mnb.trainModel()
    record.clf_end(mnb.acc_train,mnb.acc_test)
    del(mnb)

    record.clf(dtc.clf_name,dtc.getDF().shape[1])
    record.clf_start()
    dtc.trainModel()
    record.clf_end(dtc.acc_train,dtc.acc_test)
    del(dtc)

    record.clf(dtr.clf_name,dtr.getDF().shape[1])
    record.clf_start()
    dtr.trainModel()
    record.clf_end(dtr.acc_train,dtr.acc_test)
    del(dtr)

    del(record)

   
    