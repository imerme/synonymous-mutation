import os
import pickle
import time
import numpy as np
from .models import * 


def run_a_model(**params):
    #from .models import build_model
    #import tensorflow as tf
    #import keras.backend as K
    # keys: config, gpu_index, file_index, cv, epochs, batch_size, semaphore, lock, path_to_save,
    # , data
    #id_info, data, config, batch_size, epochs, gpu_id, path_to_save, lock, auc_file
    # id_info, data, config, batch_size, epochs, gpu_id, path_to_save, lock, auc_file
    gpu_index = params["gpu_index"]
    import tensorflow as tf
    from tensorflow import keras
    gpu_index = params['gpu_index']
    gpu_config = "/gpu:%d" % gpu_index
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU30%的显存
    session = tf.Session(config=config)
    params['config']['gpu_index'] = gpu_index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = params['data']
    bc = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                   min_delta=0., patience=10, verbose=2,
                                   mode='min', baseline=None)
    with tf.device(gpu_config):
        # 建立模型并训练
        model = params['model'](**params['config'])
        model.summary()
        print("正在训练config为:{}, 第{}折, 文件号为{}, 模型训练中。".format(params['config'], params['cv'], params['file_index']))
        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            batch_size=params['batch_size'],
                            epochs=params['epochs'], verbose=2, callbacks=[bc]
                            )
        print('config为:{}, 第{}折的模型训练结束。'.format(params['config'], params['cv']))
        path_to_save = params['path_to_save'] + str(params['file_index'])
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        path_to_save_model = os.path.join(path_to_save, str(params["cv"]) + ".h5")
        if params['save']:
            model.save(path_to_save_model)
        output_file_path = os.path.join(path_to_save, "output{}_cv{}.pickle".format(params['file_index'], params['cv']))
        y_pre_train = model.predict(x_train)[:, -1].tolist()
        y_pre_val = model.predict(x_val)[:, -1].tolist()
        y_pre_test = model.predict(x_test)[:, -1].tolist()
    y_train = np.array(y_train)[:, -1]
    y_test = np.array(y_test)[:, -1]
    y_val = np.array(y_val)[:, -1]
    with open(output_file_path, "ab") as f:
        pickle.dump(params['config'], f)
        pickle.dump(history.history, f)
        pickle.dump([y_pre_train, y_train.tolist()], f)
        pickle.dump([y_pre_val, y_val.tolist()], f)
        pickle.dump([y_pre_test, y_test.tolist()], f)

    del model

