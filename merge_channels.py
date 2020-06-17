import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
import pandas as pd
from core.scoring import get_scores, show_scores
from core.models import *
# tf.keras.backend.set_floatx('float32')
from itertools import combinations
from core.data import Data
from core.utils import get_kfold_index
from core.app_config import AppConfig

data = Data()
config = AppConfig()
config.CUDA_VISIBLE_DEVICES = '0'
config.data = Data()

kfold_index = get_kfold_index()  # 交叉验证的索引
cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0., patience=10, verbose=2,
                                      mode='min', baseline=None)  # 早停参数
params = dict(cv=5, epochs=500, batch_size=32, kfold_index=kfold_index, cb=[cb])
config.extent(params)

channels_names = ['scores7', 'conservation', 'sequence_feature', 'splicing']

(x_train, y_train), (x_test, y_test), (x_test_1, y_test_1), (x_test_2, y_test_2) = data.get_channels(channels_names)

 # 模型搭建
model_file_cv = [r'./models/scores7/cv{}.h5',
                 r'./models/conservation/cv{}.h5',
                 r'./models/sequence_feature/cv{}.h5',
                 r'./models/splicing/cv{}.h5']

config1 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
config2 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
config3 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
config4 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (16,)}

models_prototypes = [build_sub_model_1(**config1),
                     build_sub_model_1(**config2),
                     build_sub_model_1(**config3),
                     build_sub_model_1(**config4)]

inputs = [(7,), (7, ), (7,), (16,)]

# 用于构建结果展示的表格
first_index = []
second_index = ['train', 'validation', 'test', 'test_1', 'test_2']
tabel_content = []
columns = 'sen spe pre f1 mcc acc auc aupr tn fp fn tp'.split(' ')

counter = 0
for i in range(2, 5):
    for lst in combinations(range(4), i):
        counter += 1
        # 根据组合选择模型通道及通道配置
        x_train_comb, x_test_comb, x_test_1_comb, x_test_2_comb, model_file_cv_comb, models_prototypes_comb, inputs_comb, channels_names_comb = \
            zip(*[(x_train[indx], x_test[indx], x_test_1[indx], x_test_2[indx], model_file_cv[indx], models_prototypes[indx], inputs[indx],
                   channels_names[indx]) for indx in lst])
        lst2txt = '+'.join(channels_names_comb)
        first_index.append(lst2txt)
        file_path = r'./models/{}'.format(lst2txt)
        model_to_save = os.path.join(file_path, 'cv_{}.h5')
        # 建立保存模型的目录
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_name = file_path + '/cv{}.h5'
        train_out = []
        val_out = []
        test_out = []
        test_1_out = []
        test_2_out = []

        # 交叉验证
        pred_train = []
        pred_val = []
        pred_test = []
        pred_test_1 = []
        pred_test_2 = []
        for cv, (train_id, val_id) in enumerate(kfold_index):
            # 构建每折交叉验证的数据集
            x_train_cv, y_train_cv = [x[train_id] for x in x_train_comb], y_train[train_id]
            x_val_cv, y_val_cv = [x[val_id] for x in x_train_comb], y_train[val_id]

            # 构建模型
            model_files = [model_file.format(cv) for model_file in model_file_cv_comb]
            [model.load_weights(model_solid) for model, model_solid in zip(models_prototypes_comb, model_files)]
            model = merge_models(models_prototypes_comb, inputs_comb, **{'dp': 0.0, 'ut_1': 128, 'ut_2': 64, 'a': 'relu', 'lr': 0.00001})

            # 模型训练及保存
            model.fit(x_train_cv, y_train_cv, validation_data=(x_val_cv, y_val_cv), shuffle=True, batch_size=config.kwargs['batch_size'],
                      callbacks=config.kwargs['cb'], epochs=config.kwargs['epochs'])
            model.save(model_to_save.format(config.kwargs['cv']))

            # 模型结果处理
            y_train_pred = model.predict(x_train_cv)[:, 1]
            y_val_pred = model.predict(x_val_cv)[:, 1]
            y_test_pred = model.predict(x_test_comb)[:, 1]
            y_test_1_pred = model.predict(x_test_1_comb)[:, 1]
            y_test_2_pred = model.predict(x_test_2_comb)[:, 1]

            pred_train.append((y_train_cv[:, 1], y_train_pred))
            pred_val.append((y_val_cv[:, 1], y_val_pred))
            pred_test.append((y_test[:, 1], y_test_pred))
            pred_test_1.append((y_test_1[:, 1], y_test_1_pred))
            pred_test_2.append((y_test_2[:, 1], y_test_2_pred))

            train_out.append(get_scores(np.array(y_train_cv)[:, 1], y_train_pred, return_dict=False))
            val_out.append(get_scores(np.array(y_val_cv)[:, 1], y_val_pred, return_dict=False))
            test_out.append(get_scores(np.array(y_test)[:, 1], y_test_pred, return_dict=False))
            test_1_out.append(get_scores(np.array(y_test_1)[:, 1], y_test_1_pred, return_dict=False))
            test_2_out.append(get_scores(np.array(y_test_2)[:, 1], y_test_2_pred, return_dict=False))

            # 求交叉验证的均值并保存在tabel_content中
        file_for_out = r'./out/%d' % counter
        with open(file_for_out, 'wb') as f:
            pickle.dump(lst2txt, f)
            pickle.dump(pred_train, f)
            pickle.dump(pred_val, f)
            pickle.dump(pred_test, f)
            pickle.dump(pred_test_1, f)
            pickle.dump(pred_test_2, f)
        tabel_content.append(np.mean(train_out, axis=0))
        tabel_content.append(np.mean(val_out, axis=0))
        tabel_content.append(np.mean(test_out, axis=0))
        tabel_content.append(np.mean(test_1_out, axis=0))
        tabel_content.append(np.mean(test_2_out, axis=0))

df = pd.DataFrame(tabel_content, index=pd.MultiIndex.from_product([first_index, second_index]), columns=columns)
df.to_csv(r'./output.csv')
df.sort_values(by='auc', ascending=False).loc[(slice(None), 'validation'), :].to_csv(r'./output_validation.csv')
df.sort_values(by='auc', ascending=False).loc[(slice(None), 'test'), :].to_csv(r'./output_test.csv')

