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
matplotlib.use('Agg')

# 配置参数
config = AppConfig()
config.CUDA_VISIBLE_DEVICES = '0'
config.data = Data()

kfold_index = get_kfold_index()  # 交叉验证的索引
bc = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0., patience=10, verbose=2,
                                      mode='min', baseline=None)  # 早停参数
params = dict(cv=5, epochs=500, batch_size=32, kfold_index=kfold_index, bc=bc)
config.extent(params)

def train_a_model(data, model, output_dir, bc=config.kwargs['bc'], kfold_index=config.kwargs['kfold_index'],
                  batch_size=config.kwargs['batch_size'], epochs=config.kwargs["epochs"]):
    tr_out, val_out, te_out, h= cross_validation(data, kfold_index, model, output_dir, bc, batch_size, epochs)
    plot_cv_out(tr_out, val_out, te_out, h, output_dir)
    return val_out

# 保存输出
output_of_channels = []

# 训练子通道
channels_names = ['scores7', 'conservation', 'sequence_feature', 'splicing']
params1 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
params2 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
params3 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (7,)}
params4 = {"lr": 1e-04, "ut_1": 1024, "l1": 0.0, "ut_2": 256,
           "l2": 0.00, "dp": 0.0,'a': 'leaky_relu', 'inputs_shape': (16,)}

output_dirs = [r"./models/scores7", r'./models/conservation', r'./models/sequence_feature', r'./models/splicing']

models_prototypes = [build_sub_model_1(**params1),
                     build_sub_model_1(**params2),
                     build_sub_model_1(**params3),
                     build_sub_model_1(**params4)]

for i in range(0, 4):
    params = eval('params'+str(i+1))
    output_dir = output_dirs[i]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = models_prototypes[i]
    (x_train, y_train), (x_test, y_test), (_, _), (_, _) = config.data.get_channels([channels_names[i]])
    data = (x_train[0], y_train), (x_test[0], y_test)
    print('here', data[0][0].shape)
    output_of_channels.append(train_a_model(data, model, output_dir))


fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Blind Guess', alpha=.8)

for label, output in zip(channels_names, output_of_channels):
    line, means, std = plot_roc_curve_on_ax(ax, output)
    line.set_label(label+r' Mean ROC (area=%0.3f $\pm$ %0.3f)' % (means, std))

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel('fpr')
ax.set_ylabel('tpr')
ax.set_title('ROC of channels')
ax.legend(loc='lower left')
plt.savefig(r'./ROC curve of channels.png')
