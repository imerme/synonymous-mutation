import numpy as np
import os
from core.models import *
from core.utils import cross_validation
from core.visualization import plot_cv_out, plot_roc_curve_on_ax
import matplotlib
import matplotlib.pyplot as plt
from core.utils import get_kfold_index
from core.data import Data
from core.app_config import AppConfig
from core.scoring import scob
matplotlib.use('Agg')


param = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
         "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}

model = build_sub_model_1(**param)

data = Data()
(x_train, y_train), (x_test, y_test), (x_test_1, y_test_1), (x_test_2, y_test_2) = data.get_channels(['sequence_feature'])
model.fit(x_train[0], y_train, validation_split=0.2, batch_size=32, epochs=50)

score = scob.get_scores(y_test_2[:, 1], model.predict(x_test_2)[:, 1])
print(score)