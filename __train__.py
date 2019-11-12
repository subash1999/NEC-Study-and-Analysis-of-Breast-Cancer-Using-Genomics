
import sys,os,gc
from train_and_predict_model.chi2_train_model import Chi2TrainModel
from train_and_predict_model.relieff_train_model import ReliefFTrainModel
from train_and_predict_model.all_genes_train_model import AllGenesTrainModel
from Top_Ranking_Models.chi_square import GenomicsChiSquare
from Top_Ranking_Models.relieff import GenomicsReliefF

path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

for x in range(1,5,1):
    print("***********"*20)
    print("All Genes no of run : ",x)
    print("---------------------"*10)

    a = AllGenesTrainModel()
    a.trainAndSaveModels()
    del(a)
    gc.collect()

for x in range(1000,2000,1000):
    print("***********"*20)
    print("Chi2 Gene, no of gene : ",x)
    print("---------------------"*10)    
    t = Chi2TrainModel(x)
    t.trainAndSaveModels()
    del(t)
    gc.collect()

for x in range(1000,5000,1000):
    for y in range(20,30,10):
        print("***********"*20)
        print("ReliefF Gene, no of gene : ",x," AND neighbors : ",y)
        print("---------------------"*10)   
        t = ReliefFTrainModel(x,y)
        t.trainAndSaveModels()
        del(t)
        gc.collect()
