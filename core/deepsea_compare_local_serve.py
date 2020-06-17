from utils import get_from_deepsea
import os
os.chdir(r'F:/SNPP')

(x_train_net, y_train), (x_test_net, y_test) = get_from_deepsea(columns='log', dir_name=r'./data/raw data/deepsea_features')
(x_train_local, y_train_l), (x_test_local, y_test_l) = get_from_deepsea(columns='log', dir_name=r'./data/raw data/local_deepsea_features')

