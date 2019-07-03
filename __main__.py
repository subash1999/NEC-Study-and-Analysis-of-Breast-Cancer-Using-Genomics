# from Top_Ranking_Models.chi_square import GenomicsChiSquare

# from Classification_Models.support_vector_machine import GenomicsSVC
# from Classification_Models.linear_regression import GenomicsLR
# from Classification_Models.k_nearest_neighbour import GenomicsKNN
# from Classification_Models.k_means import GenomicsKMeans
# from Classification_Models.bernouli_nb import GenomicsBNB
# from Classification_Models.complement_nb import GenomicsCNB
# from Classification_Models.gaussian_nb import GenomicsGNB
# from Classification_Models.multinomial_nb import GenomicsMNB
# from Classification_Models.decision_tree_classification import GenomicsDTC
# from Classification_Models.decision_tree_regression import GenomicsDTR

import sys, os
path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)
print(sys.path)
