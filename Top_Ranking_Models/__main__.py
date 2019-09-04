import sys, os
path = os.path.join(os.path.dirname(__file__),'..')
if path not in sys.path:
    sys.path.append(path)

# from chi_square import GenomicsChiSquare

# g = GenomicsChiSquare()


from relieff import GenomicsReliefF
r = GenomicsReliefF()
r.selectTopGenes(r.x,r.y)