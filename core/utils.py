import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing as mp
from itertools import product
from .scoring import scob
import json

def cross_validation(data, kfold_index, model, model_path,cb, batch_size=32, epochs=500):
    # 对模型进行交叉验证和保存模型
    (x_train, y_train), (x_test, y_test) = data
    initial_model = os.path.join(model_path, "initial_model.h5")
    if not os.path.isfile(initial_model):
        model.save(initial_model)
    train_out = []
    val_out = []
    test_out = []
    historys = []
    for cv, (train_id, val_id) in enumerate(kfold_index):
        try:
            x_train_cv, y_train_cv = x_train[train_id], y_train[train_id]
            x_val_cv, y_val_cv = x_train[val_id], y_train[val_id]
        except:
            x_train_cv, y_train_cv = [x[train_id] for x in x_train], y_train[train_id]
            x_val_cv, y_val_cv = [x[val_id] for x in x_train], y_train[val_id]

        model.load_weights(initial_model)
        history = model.fit(x_train_cv, y_train_cv, validation_data=(x_val_cv, y_val_cv), shuffle=True, callbacks=[cb],
                  batch_size=batch_size, epochs=epochs)
        if model_path:
            model.save(os.path.join(model_path, 'cv{}.h5'.format(cv)))
        y_train_pred = model.predict(x_train_cv)[:, 1]
        y_val_pred = model.predict(x_val_cv)[:, 1]
        y_test_pred = model.predict(x_test)[:, 1]

        train_out.append((y_train_cv[:, 1], y_train_pred))
        val_out.append((y_val_cv[:, 1], y_val_pred))
        test_out.append((y_test[:, 1], y_test_pred))
        historys.append(history.history)

    return train_out, val_out, test_out, historys


def get_data_by_channels_padding_zero(channels=[], random_state=1, shuffle=True, data_dir=''):
    train_index=np.arange(2396)
    if shuffle:
        nrs = np.random.RandomState(random_state)
        nrs.shuffle(train_index)
    x_train_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/1198train.txt'))[:, 1:][train_index]
    y_train_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/1198train.txt'))[:, 0][train_index]
    y_train_base = np.eye(2)[y_train_base.astype(int)]

    x_test_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/96test.txt'))[:, 1:]
    y_test_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/96test.txt'))[:, 0]
    y_test_base = np.eye(2)[y_test_base.astype(int)]

    x_test_1_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/HGMDtest.txt'))[:, 1:]
    y_test_1_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/HGMDtest.txt'))[:, 0]
    y_test_1_base = np.eye(2)[y_test_1_base.astype(int)]

    x_test_2_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/IDSVtest.txt'))[:, 1:]
    y_test_2_base = np.loadtxt(os.path.join(data_dir, 'data/0 padding/IDSVtest.txt'))[:, 0]
    y_test_2_base = np.eye(2)[y_test_2_base.astype(int)]

    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)
    log = slice(40, 40 + 919)
    diff = slice(40 + 919, 40 + 919 + 919)
    ref = slice(40 + 919 + 919, 40 + 919 + 919 + 919)
    alt = slice(40 + 919 + 919 + 919, None)

    x_train = []
    x_test = []
    x_test_1 = []
    x_test_2 = []

    for channel in channels:
        x_train.append(x_train_base[:, eval(channel)])
        x_test.append(x_test_base[:, eval(channel)])
        x_test_1.append(x_test_1_base[:, eval(channel)])
        x_test_2.append(x_test_2_base[:, eval(channel)])
    return (x_train, y_train_base), (x_test, y_test_base), (x_test_1, y_test_1_base), (x_test_2, y_test_2_base)

def get_data_by_channels(channels=[], random_state=1, shuffle=True, data_dir=''):
    # 根据通道名 获取数据
    train_index = np.arange(2396)
    if shuffle:
        nrs = np.random.RandomState(random_state)
        nrs.shuffle(train_index)
    y_train = np.eye(2)[np.r_[np.ones((2396 // 2)), np.zeros((2396 // 2))][train_index].astype(int)]
    y_test = np.eye(2)[np.r_[np.ones((192 // 2)), np.zeros((192 // 2))].astype(int)]


    if data_dir:
        x_train_base = np.load(os.path.join(data_dir, 'data/cooked data/x_train_plus.npy'))[train_index]
        x_test_base = np.load(os.path.join(data_dir, 'data/cooked data/x_test_plus.npy'))
    else:
        x_train_base = np.load(r'./data/cooked data/x_train_plus.npy')[train_index]
        x_test_base = np.load(r'./data/cooked data/x_test_plus.npy')

    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)
    log = slice(40, 40+919)
    diff = slice(40+919, 40+919+919)
    ref = slice(40+919+919, 40+919+919+919)
    alt = slice(40+919+919+919, None)

    x_train = []
    x_test = []

    for channel in channels:
        if channel in ['log', 'diff', 'alt', 'ref']:
            x_train.append(x_train_base[:, eval(channel)].reshape(-1, 919, 1))
            x_test.append(x_test_base[:, eval(channel)].reshape(-1, 919, 1))
        else:
            x_train.append(x_train_base[:, eval(channel)])
            x_test.append(x_test_base[:, eval(channel)])
    return (x_train, y_train), (x_test, y_test)
    # 获取训练集数据

def get_data_by_channels_before_sorted(channels=[], random_state=1):
    # 根据通道名 获取数据
    nrs = np.random.RandomState(random_state)
    train_index = np.arange(2396)
    nrs.shuffle(train_index)
    y_train = np.eye(2)[np.r_[np.ones((2396 // 2)), np.zeros((2396 // 2))][train_index].astype(int)]
    y_test = np.eye(2)[np.r_[np.ones((192 // 2)), np.zeros((192 // 2))].astype(int)]

    x_train_base = np.load(r'./data/cooked data/x_train_before.npy')[train_index]
    x_test_base = np.load(r'./data/cooked data/x_test_before.npy')

    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)
    log = slice(40, 40 + 919)
    diff = slice(40 + 919, 40 + 919 + 919)

    x_train = []
    x_test = []

    for channel in channels:
        if channel in ['log', 'diff']:
            x_train.append(x_train_base[:, eval(channel)].reshape(-1, 919, 1))
            x_test.append(x_test_base[:, eval(channel)].reshape(-1, 919, 1))
        else:
            x_train.append(x_train_base[:, eval(channel)])
            x_test.append(x_test_base[:, eval(channel)])
    return (x_train, y_train), (x_test, y_test)
    # 获取错误的训练集数据


def get_params_iter(dictionary):
    for params in product(*dictionary.values()):
        yield dict(zip(dictionary.keys(), params))
        # 产生网格搜索时的迭代字典

def get_data_by_channels_before_sorted2(channels=[], random_state=1):
    # 根据通道名 获取数据
    nrs = np.random.RandomState(random_state)
    train_index = np.arange(2396)
    nrs.shuffle(train_index)
    y_train = np.eye(2)[np.r_[np.ones((2396 // 2)), np.zeros((2396 // 2))][train_index].astype(int)]
    y_test = np.eye(2)[np.r_[np.ones((192 // 2)), np.zeros((192 // 2))].astype(int)]

    x_train_base = np.load(r'./data/cooked data/x_train_before2.npy')[train_index]
    x_test_base = np.load(r'./data/cooked data/x_test_before2.npy')

    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)
    log = slice(40, 40 + 919)
    diff = slice(40 + 919, 40 + 919 + 919)

    x_train = []
    x_test = []

    for channel in channels:
        if channel in ['log', 'diff']:
            x_train.append(x_train_base[:, eval(channel)].reshape(-1, 919, 1))
            x_test.append(x_test_base[:, eval(channel)].reshape(-1, 919, 1))
        else:
            x_train.append(x_train_base[:, eval(channel)])
            x_test.append(x_test_base[:, eval(channel)])
    return (x_train, y_train), (x_test, y_test)
    # 获取错误的训练集数据（有问题 勿用）

def cross_validation_merge(data, kfold, model, model_path, cv, batch_size=32, epochs=500, ):
    # wrong  (勿用)
    pass


def gdbc_independent(no=1, sorted=True, channels=[]):
    # 获取独立测试集数据
    if no == 1:
        y_test = np.eye(2)[np.r_[np.ones((246)), np.zeros((246))].astype(int)]
        if sorted:
            x_test_base = np.load(r'data/cooked data/independent_HGMD_VariSNP.npy')
        else:
            x_test_base = np.load(r'data/cooked data/independent_HGMD_VariSNP_bs.npy')
    elif no == 2:
        y_test = np.eye(2)[np.r_[np.ones((93)), np.zeros((5039))].astype(int)]
        if sorted:
            x_test_base = np.load(r'data/cooked data/independent_IDSV_ClinVar.npy')
        else:
            x_test_base = np.load(r'data/cooked data/independent_IDSV_ClinVar_bs.npy')
    else:
        y_test = np.eye(2)[np.r_[np.ones((1088)), np.zeros((1088))].astype(int)]
        if sorted:
            x_test_base = np.load(r'data/cooked data/independent_SynMICdb.npy')
        else:
            x_test_base = np.load(r'data/cooked data/independent_SynMICdb_bs.npy')



    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)
    log = slice(40, 40 + 919)
    diff = slice(40 + 919, 40 + 919 + 919)

    x_test = []

    for channel in channels:
        if channel in ['log', 'diff']:
            x_test.append(x_test_base[:, eval(channel)].reshape(-1, 919, 1))
        else:
            x_test.append(x_test_base[:, eval(channel)])
    return x_test, y_test
    # 获取独立验证集数据，no = 1 独立测试集1.

# def get_sequence_data(shuffle=True, random_state = 0, one_hot = True):
#     # 获取序列文件
#     # 默认打乱迅速
#     nr = np.random.RandomState(random_state)
#     train_index = np.arange(2396)
#     if shuffle:
#         nr.shuffle(train_index)
#
#     train_alt = pd.read_csv(r'../data/cooked_data/train_alt.csv').iloc[train_index, -1]
#     train_ref = pd.read_csv(r'../data/cooked_data/train_ref.csv').iloc[train_index, -1]
#     test_alt = pd.read_csv(r'../data/cooked_data/test_alt.csv').iloc[:, -1]
#     test_ref = pd.read_csv(r'../data/cooked_data/test_ref.csv').iloc[:, -1]
#     if one_hot:
#         for data in [train_alt, train_ref, test_alt, test_ref]:
#             pd.apply(onehot, axis=0, data)
#
#     y_train = np.eye(2)[np.r_[np.ones((2396 // 2)), np.zeros((2396 // 2))][train_index].astype(int)]
#     y_test = np.eye(2)[np.r_[np.ones((192 // 2)), np.zeros((192 // 2))].astype(int)]
#
#     return (train_ref, train_alt, y_train), (test_ref, test_alt, y_test)

def get_raw_seq():
    x_train_alt = pd.read_csv(r'data/cooked data/train_alt.csv')
    x_train_ref = pd.read_csv(r'data/cooked data/train_ref.csv')
    x_test_alt = pd.read_csv(r'data/cooked data/test_alt.csv')
    x_test_ref = pd.read_csv(r'data/cooked data/test_ref.csv')
    return (x_train_ref, x_train_alt), (x_test_ref, x_test_alt)

def get_cooked_seq(by='codon', length=50):
    data = get_raw_seq()
    (train_ref, train_alt), (test_ref, test_alt) = data

    data_dict = {}
    for dataset, key in zip([train_ref, train_alt, test_ref, test_alt], ['train_ref', 'train_alt', 'test_ref', 'test_alt']):
        dataset.pos_info = dataset.pos_info.str.extract('(?<=c.)(\d+)(?=\w>\w)')
        data_dict[key] = []
        for row in dataset.itertuples():
            data_dict[key].append(onehot(cut_seq(int(row.pos_info), row.DNA_sequence,by=by, length=length)))
        data_dict[key] = np.array(data_dict[key])
    return data_dict


def onehot(seq, depth=5, dict_map = {'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}):
    e = np.eye(depth)
    dict_map = dict_map
    return e[[i for i in map(lambda x: dict_map[x], seq)]]

def cut_seq(pos, seq, length=50, by='codon'):
    codon_start = pos//3 * 3
    codon_end = pos//3*3 +2
    if by=='codon':
        start = codon_start - length*3
        end = codon_end + length*3
    else:
        start = pos - length
        end = pos + length
    if start >= 0 and end <= len(seq)-1:
        return seq[start: end+1]
    elif start <0 and end <= len(seq)-1:
        return abs(start)*'N' + seq[: end+1]
    elif start >=0 and end > len(seq)-1:
        return seq[start:] + 'N'*(end-len(seq)+1)
    else:
        return abs(start)*'N' + seq + 'N'*(end-len(seq)+1)



def get_796data_by_channels(channels=[], shuffle=True, random_state=1):
    # 打乱数据
    # 打乱训练集数据的索引
    train_index = np.arange(1592)
    if shuffle:
        nrs = np.random.RandomState(random_state)
        nrs.shuffle(train_index)
    y_train = np.eye(2)[np.r_[np.ones(( 1592// 2)), np.zeros((1592 // 2))][train_index].astype(int)]
    y_test = np.eye(2)[np.r_[np.ones((83,)), np.zeros((83,))].astype(int)]
    y_test_clinvar = np.eye(2)[np.r_[np.ones((56,)), np.zeros((78,))].astype(int)]
    y_test_hgmd = np.eye(2)[np.r_[np.ones((159,)), np.zeros((159,))].astype(int)]

    # 获取独立测试集数据
    x_train_40_base = np.load(r'./data/cooked data/x_train_796.npy')[train_index]
    x_test_40_base = np.load(r'./data/cooked data/x_test_796.npy')
    x_test_clinvar_base = np.load(r'./data/cooked data/x_test_clinvar_796.npy')
    x_test_hgmd_base = np.load(r'./data/cooked data/x_test_hgmd_796.npy')

    columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
    sequence_feature = ['DSP', 'CpG?', 'CpG_exon',
                        'f_premrna', 'f_mrna', 'RSCU',
                        'dRSCU']
    splicing = ['MES', 'dMES', 'MES+', 'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?',
                'SR-', 'SR+', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+',
                'PESS-', 'PESS+', 'dPSIZ']
    conservation = ['priPhyloP', 'mamPhyloP', 'verPhyloP', 'priPhCons', 'mamPhCons', 'verPhCons', 'GerpS']
    function_regions_annotation = ['TFBs']
    trainslation_efficiency = ['TE']
    sequence_feature = [columns.index(column) for column in sequence_feature]
    splicing = [columns.index(column) for column in splicing]
    conservation = [columns.index(column) for column in conservation]
    function_regions_annotation = [columns.index(column) for column in function_regions_annotation]
    trainslation_efficiency = [columns.index(column) for column in trainslation_efficiency]
    scores7 = slice(1, 8)
    scores8 = slice(0, 8)
    scores32 = slice(8, 40)
    scores40 = slice(0, 40)

    x_train = []
    x_test = []
    x_test_hgmd = []
    x_test_clinvar = []
    for channel in channels:
            x_train.append(x_train_40_base[:, eval(channel)])
            x_test.append(x_test_40_base[:, eval(channel)])
            x_test_hgmd.append(x_test_hgmd_base[:, eval(channel)])
            x_test_clinvar.append(x_test_clinvar_base[:, eval(channel)])

    return (x_train, y_train), (x_test, y_test),  (x_test_hgmd, y_test_hgmd), (x_test_clinvar, y_test_clinvar)


def get_kfold_index(n=1198*2):
    np.random.seed(1)  # 种子保持不变
    y_train = np.arange(n)
    kfold_index = []
    kfold = KFold(n_splits=5, shuffle=False, random_state=1).split(y_train)
    for index in kfold:
        kfold_index.append(index)
    return kfold_index

def cross_validation_v2(data, kfold_index, build_func, params, model_path,cb, batch_size=32, epochs=500):
    # 对模型进行交叉验证和保存模型
    (x_train, y_train), (x_test, y_test) = data
    train_out = []
    val_out = []
    test_out = []
    historys = []
    for cv, (train_id, val_id) in enumerate(kfold_index):
        print(x_train.shape, y_train.shape)
        try:
            x_train_cv, y_train_cv = x_train[train_id], y_train[train_id]
            x_val_cv, y_val_cv = x_train[val_id], y_train[val_id]
        except:
            x_train_cv, y_trai = [x[train_id] for x in x_train], y_train[train_id]
            x_val_cv, y_val_cv = [x[val_id] for x in x_train], y_train[val_id]
        model = build_func(**params)
        history = model.fit(x_train_cv, y_train_cv, validation_data=(x_val_cv, y_val_cv), shuffle=True, callbacks=[cb],
                  batch_size=batch_size, epochs=epochs)
        if model_path:
            model.save(os.path.join(model_path, 'cv{}.h5'.format(cv)))

        y_train_pred = model.predict(x_train_cv)[:, 1]
        y_val_pred = model.predict(x_val_cv)[:, 1]
        y_test_pred = model.predict(x_test)[:, 1]

        train_out.append((y_train_cv[:, 1], y_train_pred))
        val_out.append((y_val_cv[:, 1], y_val_pred))
        test_out.append((y_test[:, 1], y_test_pred))
        historys.append(history.history)
    return train_out, val_out, test_out, historys


def indicator(best_index, best_params, file_indicator):
    best_params = json.dumps(best_params)
    best_index = str(best_index)
    with open(file_indicator, 'w') as f:
        f.write('最好的参数序号：' + best_index +'\n')
        f.write('参数：'+ best_params)
        
    print('end')

def solid_outcome(train_score, val_score, test_score, params, i, file_to_save):
    train_score = json.dumps(train_score.tolist())
    val_score = json.dumps(val_score.tolist())
    test_score = json.dumps(test_score.tolist())
    params = json.dumps(params)
    i = str(i)
    with open(file_to_save, 'a+') as f:
        f.write(i+ '\t' + params + '\t' + 'train_score' + '\t' + train_score)
        f.write(i + '\t' + params + '\t' + 'val_score' + '\t' + val_score)
        f.write(i + '\t' + params + '\t' + 'test_score' + '\t' + test_score)

def gridsearch(build_model_func, config_dict, data, cv_index, cb,
               batch_size, epochs, model_path=None,
               file_to_save=r'./scores_of_gscv.csv',
               file_indicator=r'./indicator.txt'):
    best_auc = 0
    best_params = {}

    for i, params in enumerate(get_params_iter(config_dict)):
        train_out, val_out, test_out, _ = cross_validation_v2(data=data, kfold_index=cv_index,
                                                           build_func=build_model_func,
                                                           model_path=model_path,
                                                           params=params,
                                                           cb=cb,
                                                           batch_size=batch_size,
                                                           epochs=epochs)

        train_score = scob.get_scores_from_cv_out(train_out, return_mean_only=True)
        val_score = scob.get_scores_from_cv_out(val_out, return_mean_only=True)
        test_score = scob.get_scores_from_cv_out(val_out, return_mean_only=True)

        # 保存训练集验证集测试集得分
        p = mp.Process(target=solid_outcome,
                       args=[train_score, val_score, test_score, params, i, file_to_save])
        p.start()

        # 获得最好的参数
        val_auc = val_score[0,6]

        if val_auc > best_auc:
            best_params = params
            best_index = i
            q = mp.Process(target=indicator, args=[best_index, best_params, file_indicator])
            q.start()
