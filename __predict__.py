import pandas as pd
import numpy as np
import sys,os,gc
path = os.path.join(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

from train_and_predict_model.predict import Predict
p = Predict()

# p.showOptions()