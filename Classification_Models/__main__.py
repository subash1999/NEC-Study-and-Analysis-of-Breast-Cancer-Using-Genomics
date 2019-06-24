from support_vector_machine import GenomicsSVC
from linear_regression import GenomicsLR
from k_nearest_neighbour import GenomicsKNN
from k_means import GenomicsKMeans
from bernouli_nb import GenomicsBNB
from complement_nb import GenomicsCNB
from gaussian_nb import GenomicsGNB
from multinomial_nb import GenomicsMNB
from decision_tree_classification import GenomicsDTC
from decision_tree_regression import GenomicsDTR


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
