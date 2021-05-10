import os,re
import errno
import random
import urllib.request as urllib
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle

def fliter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match( r'(.*)FE_time', key, re.M|re.I)
        if matchObj:
            fkeys.append(matchObj.group(1))
    if(len(fkeys)>1):
        print(keys)
    return fkeys[0]+'DE_time',fkeys[0]+'FE_time'


exps_idx = {
    '12DriveEndFault':0,
    '12FanEndFault':9,
    '48DriveEndFault':0
}

faults_idx = {
    'Normal': 0,
    '0.007-Ball': 1,
    '0.014-Ball': 2,
    '0.021-Ball': 3,
    '0.007-InnerRace': 4,
    '0.014-InnerRace': 5,
    '0.021-InnerRace': 6,
    '0.007-OuterRace6': 7,
    '0.014-OuterRace6': 8,
    '0.021-OuterRace6': 9,
#     '0.007-OuterRace3': 10,
#     '0.014-OuterRace3': 11,
#     '0.021-OuterRace3': 12,
#     '0.007-OuterRace12': 13,
#     '0.014-OuterRace12': 14,
#     '0.021-OuterRace12': 15,
}

def get_class(exp,fault):
    if fault == 'Normal':
        return 0
    return exps_idx[exp] + faults_idx[fault]
    
 
class CWRU:
    def __init__(self, exps, rpms, length):
        for exp in exps:
            if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
                print("wrong experiment name: {}".format(exp))
                return
        for rpm in rpms:    #下方的转速可以看做是载荷
            if rpm not in ('1797', '1772', '1750', '1730'):
                print("wrong rpm value: {}".format(rpm))
                return
        # root directory of all data
        rdir = os.path.join('Datasets/CWRU')
        print(rdir,exp,rpm)

        fmeta = os.path.join(os.path.dirname('__file__'), 'metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            # 0 = '12DriveEndFault'
            # 1 = '1730'
            # 2 = '0.007-Ball'
            # 3 = 'http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/121.mat'

            if (l[0] in exps or l[0] == 'NormalBaseline') and l[1] in rpms:
                if 'Normal' in l[2] or '0.007' in l[2] or '0.014' in l[2] or '0.021' in l[2]:
                    if faults_idx.get(l[2],-1)!=-1:
                        lines.append(l)
 
        self.length = length  # sequence length
        lines = sorted(lines, key=lambda line: get_class(line[0],line[2])) 
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.all_labels = tuple(((line[0]+line[2]),get_class(line[0],line[2])) for line in lines)
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1]) 
        self.nclasses = len(self.classes)  # number of classes
 
    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)
 
    def _download(self, fpath, link):
        print(link + " Downloading to: '{}'".format(fpath))
        urllib.urlretrieve(link, fpath)

    def _load_and_slice_data(self, rdir, infos):
        """

        :param rdir: Datasets/CWRU 表示数据的路面
        :param infos: 驱动端故障和正常状态的所有数据
        :return:
        """
        #self.X_train = np.zeros((0, self.length, 2))
        #self.X_test = np.zeros((0, self.length, 2))
        self.X_train = []
        self.X_test =[]
        self.y_train = []
        self.y_test = []
        # star: 计数从star开始.默认是从0开始.
        # stop: 计数到stop结束,但不包括stop.
        # step: 步长,默认为80.  然后取 前660个
        train_cuts = list(range(0,60000,80))[:660]
        test_cuts = list(range(60000,120000,self.length))[:25]
        for idx, info in enumerate(infos):
 
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            print(idx,fpath)
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))
 
            mat_dict = loadmat(fpath)
            key1,key2 = fliter_key(mat_dict.keys())
            #<class 'tuple'>: (485643, 2)  485643是存储的所有数据个数
            time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
            idx_last = -(time_series.shape[0] % self.length)
            print(time_series.shape)
            #先让上一个idx_last使用shape[0] 然后再转list
            time_series=time_series.tolist()

            #这个shape 是0 2 所以是没有数据的 只有shape 为（0,2)
            clips = []
            shuffle_get=shuffle(train_cuts)
            for cut in shuffle_get:
                # 按垂直方向（行顺序）堆叠数组
                #clips = np.vstack((clips, time_series[cut:cut+self.length]))
                clips.append(time_series[cut:cut+self.length])

            self.X_train.append( clips)

            clips = []
            for cut in shuffle(test_cuts):
                #clips = np.concatenate((clips, time_series[cut:cut+self.length]))
                clips.append(time_series[cut:cut+self.length])

            self.X_test.append(clips)
            
            self.y_train += [get_class(info[0],info[2])] * 660
            self.y_test += [get_class(info[0],info[2])] * 25

        self.X_train=np.array(self.X_train)
        self.X_test=np.array(self.X_test)

        self.X_train=self.X_train.reshape(-1, self.length,2)
        self.X_test=self.X_test.reshape(-1, self.length,2)
 
    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))
 
        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        # 这里的shape0 应该是750的
        self.y_test = np.array(tuple(self.y_test[i] for i in index))
        random.Random(0).shuffle(index)
if __name__=='__main__':
    window_size = 2048
    #大部分论文好像都是驱动端的故障
    data = CWRU(['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
    # np.save("exp3_data.npy", data)
    # data = np.load("dataNumpy.npz")
    # #data = np.load("exp3_data.npy")

    print(data.nclasses, data.classes, len(data.X_train), len(data.X_test))