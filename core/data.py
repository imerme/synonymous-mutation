# amira
# 20200616
# rainy
import os
import numpy as np

columns = "priPhCons	mamPhCons	verPhCons	priPhyloP	mamPhyloP	verPhyloP	GerpS	TFBs	TE	dPSIZ	DSP	RSCU	dRSCU	CpG?	CpG_exon	SR-	SR+	FAS6-	FAS6+	MES	dMES	MES+	MES-	MEC-MC?	MEC-CS?	MES-KM?	PESE-	PESE+	PESS-	PESS+	f_premrna	f_mrna".split(
        '\t')
sequence_feature = ['DSP', 'CpG?', 'CpG_exon', 'f_premrna', 'f_mrna', 'RSCU', 'dRSCU']
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
# log = slice(40, 40+919)
# diff = slice(40+919, 40+919+919)
# ref = slice(40+919+919, 40+919+919+919)
# alt = slice(40+919+919+919, None)


class Data(object):
    """
    训练集： (x_train, y_train) 1196+1196
    测试集： (x_test, y_test) 96+96
    独立测试集1: (x_test_1, y_test_1) 246 + 246
    独立测试集2： （x_test_2, y_test_2） 93 + 5039
    """
    def __init__(self, data_dir=r'./data/cooked data'):
        self.data_dir = data_dir
        self._get_data()

    def _get_data(self):
        # 私有方法，获取数据
        self.x_train = np.load(os.path.join(self.data_dir, 'x_train.npy'))
        self.y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
        self.x_test = np.load(os.path.join(self.data_dir, 'x_test.npy'))
        self.y_test = np.load(os.path.join(self.data_dir, 'y_test.npy'))
        self.x_test_1 = np.load(os.path.join(self.data_dir, 'x_test_1.npy'))
        self.y_test_1 = np.load(os.path.join(self.data_dir, 'y_test_1.npy'))
        self.x_test_2 = np.load(os.path.join(self.data_dir, 'x_test_2.npy'))
        self.y_test_2 = np.load(os.path.join(self.data_dir, 'y_test_2.npy'))

    def _get_channels(self, x_data, y_data, channels):
        # 私有方法，根据通道获取数据
        channels_data = [x_data[:, eval(channel)] for channel in channels]
        return channels_data, y_data

    def get_channels(self, channels, shuffle=True):
        out_data = []
        if shuffle:
            index = np.arange(len(self.x_train))
            np.random.seed(0)
            np.random.shuffle(index)
            x_train = self.x_train[index]
            y_train = self.y_train[index]
        else:
            x_train = self.x_train
            y_train = self.y_train
        for x, y in [(x_train, y_train), (self.x_test, self.y_test), (self.x_test_1, self.y_test_1),
                     (self.x_test_2, self.y_test_2)]:
            out_data.append(self._get_channels(x, y, channels))
        return out_data

