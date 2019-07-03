import sys, os
path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)
# print(sys.path)

from Classification_Models.support_vector_machine import GenomicsSVC
from Top_Ranking_Models.chi_square import GenomicsChiSquare
import pandas as pd 
import numpy as np

from record_performance import RecordPeformance

record = RecordPeformance("test")

chi2 = GenomicsChiSquare()
svc  = GenomicsSVC()

record.tot(svc.clf_name,100,chi2.method_name)



record.top(chi2.method_name,100)
record.top_start()
chi2.makeTopGenesDF()
df_top = chi2.getChi2TopGenesDF()
record.top_end()

print(df_top)

record.clf(svc.clf_name,100)
record.clf_start()
svc.setDF(df_top)
svc.trainModel()
record.clf_end(svc.acc_train,svc.acc_test)

record.tot_end(svc.acc_train,svc.acc_test)