import sys, os
path = os.path.join(os.path.dirname(__file__),'..')
if path not in sys.path:
    sys.path.append(path)
print(sys.path)

from support_vector_machine import GenomicsSVC
from linear_regression import GenomicsLR
from k_nearest_neighbour import GenomicsKNN
from k_means import GenomicsKMeans
from decision_tree import GenomicsDTC,GenomicsDTR
from naives_bayes import GenomicsBNB,GenomicsBNB,GenomicsGNB,GenomicsMNB,GenomicsNB

print("*"*8,"\t\t Classification : ","*"*8)
g = GenomicsDTC()
g.trainModel()


print("*"*8,"\t\t Regression : ","*"*8)
g = GenomicsDTR()
g.trainModel()
# print("\nSupport Vector Machine")
# g = GenomicsKMeans()
# g.trainModel()
# print(g.model.labels_)
# c_1 = 0
# c_0 = 0
# for val in g.model.labels_:
#     if val == 1 : 
#         c_1 +=1
#     else:
#         c_0 +=1
# print(len(g.model.labels_))
# print('C 1 :',c_1)

# print('C 0 :',c_0)
