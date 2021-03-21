
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
# set the memory usage
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import imp
import pandas as pd

#
import cwru

window_size = 2048
data = cwru.CWRU(['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
print("data.nclasses:",data.nclasses,"data.classes:",data.classes)
print("len(data.X_train):",len(data.X_train),"len(data.X_test):",len(data.X_test))

#
import models
# # imp.reload(models)
# siamese_net = models.load_siamese_net((window_size,2))
# print('\nsiamese_net summary:')
# siamese_net.summary()
# #
# print('\nsequential_3 is WDCNN:')
# siamese_net.layers[2].summary()
# #
# wdcnn_net = models.load_wdcnn_net()
# print('\nwdcnn_net summary:')
# wdcnn_net.summary()


#
import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping

import siamese
imp.reload(siamese)
import utils
imp.reload(utils)

snrs = [-4,-2,0,2,4,6,8,10,None]


settings = {
  "N_way": 10,           # how many classes for testing one-shot tasks>
  "batch_size": 32,
  "best": -1,
  "evaluate_every": 200,   # interval for evaluating on one-shot tasks
  "loss_every": 20,      # interval for printing loss (iterations)
  "n_iter": 15000,
  "n_val": 2,          #how many one-shot tasks to validate on?
  "n": 0,
  "save_path":"",
  "save_weights_file": "weights-best-10-oneshot-low-data.hdf5"
}

exp_name = "EXP-AB"
exps = [60,90,120,200,300,600,900,1500,6000]
#exps = [1980,6000]
times = 20
#缺少了1500 第9次
is_training = True  # enable or disable train models. if enable training, save best models will be update.


def EXPAB_train_and_test(exp_name, exps, is_training):
    train_classes = sorted(list(set(data.y_train)))
    train_indices = [np.where(data.y_train == i)[0] for i in train_classes]
    for exp in exps: #总的训练样本数
        scores_1_shot = []
        scores_5_shot = []
        scores_5_shot_prod = []
        scores_wdcnn = []
        num = int(exp / len(train_classes))
        settings['evaluate_every'] = 300 if exp < 1000 else 600
        print("settings['evaluate_every']:",settings['evaluate_every'])
        for time_idx in range(10):  #重复20次，每次的随机种子不一样
            seed = int(time_idx / 4) * 10
            np.random.seed(seed)
            print('random seed:', seed)
            print("\n样本数%s-第%s次训练" % (exp, time_idx) + '*' * 80)
            #路径：temp/EXP-AB/size_exp/time_time_idx
            settings["save_path"] = "tmp/%s/size_%s/time_%s/" % (exp_name, exp, time_idx)
            data._mkdir(settings["save_path"])

            train_idxs = []
            val_idxs = []
            for i, c in enumerate(train_classes):
                select_idx = train_indices[i][np.random.choice(len(train_indices[i]), num, replace=False)]
                split = int(0.6 * num) #使用60%样本作为训练集
                train_idxs.extend(select_idx[:split])
                val_idxs.extend(select_idx[split:])
            X_train, y_train = data.X_train[train_idxs], data.y_train[train_idxs],
            X_val, y_val = data.X_train[val_idxs], data.y_train[val_idxs],

            print("训练集前10个元素的下标:",train_idxs[0:10])
            print("验证集前10个元素的下标:",val_idxs[0:10])

            # load one-shot model and training  修改的
            siamese_net = models.load_siamese_net_my_mew()
            siamese_loader = siamese.Siamese_Loader(X_train,
                                                    y_train,
                                                    X_val,
                                                    y_val)

            if (is_training):
                print(siamese.train_and_test_oneshot(settings, siamese_net, siamese_loader))

            # 将具体类别数转换为  向量化
            y_train = keras.utils.to_categorical(y_train, data.nclasses)
            y_val = keras.utils.to_categorical(y_val, data.nclasses)
            y_test = keras.utils.to_categorical(data.y_test, data.nclasses)

            earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
            # checkpoint
            # filepath="tmp/weights-best-cnn-{epoch:02d}-{val_acc:.2f}.hdf5"
            filepath = "%sweights-best-10-cnn-low-data.hdf5" % (settings["save_path"])
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [earlyStopping, checkpoint]

            wdcnn_net = models.load_wdcnn_net()
            if (is_training):
                wdcnn_net.fit(X_train, y_train,
                              batch_size=32,
                              epochs=300,
                              verbose=0,
                              callbacks=callbacks_list,
                              validation_data=(X_val, y_val))

            # loading best weights and testing
            print("load best weights", settings["save_path"] + settings['save_weights_file'])
            siamese_net.load_weights(settings["save_path"] + settings['save_weights_file'])
            print("load best weights", filepath)
            wdcnn_net.load_weights(filepath)
            for snr in snrs:
                print("\n样本数%s_第%s次训练_噪音为%s" % (exp, time_idx, snr) + '*' * 80)
                X_test_noise = []
                if snr != None:
                    for x in data.X_test:
                        X_test_noise.append(utils.noise_rw(x, snr))
                    X_test_noise = np.array(X_test_noise)
                else:
                    X_test_noise = data.X_test

                # test 1_shot and 5_shot
                siamese_loader.set_val(X_test_noise, data.y_test)
                s = 'val'
                preds_5_shot = []
                prods_5_shot = []
                scores = []
                for k in range(5):
                    #val_acc 是一个ontshot的正确率，preds是预测类别和真实类别的组合
                    #prods是 预测某一个样本在 10个类别上的概率
                    val_acc, preds, prods = siamese_loader.test_oneshot2(siamese_net, N=len(siamese_loader.classes[s]),
                                                                         k=len(siamese_loader.data[s]), verbose=False)
                    #                 utils.confusion_plot(preds[:,1],preds[:,0])
                    print("测试集的正确率:",val_acc, preds.shape, prods.shape)
                    scores.append(val_acc)
                    preds_5_shot.append(preds[:, 1])   #list 每个是ndarray  shape=750
                    prods_5_shot.append(prods) #list 每个是ndarray  shape=750 10  1
                preds = []
                for line in np.array(preds_5_shot).T:
                    pass
                    #np.argmax(np.bincount(line)) 就是 一个样本 5次ont-shot 中选择出现次数最多的
                    #预测类别比如54555  那么就选5 ，代表是第5类
                    preds.append(np.argmax(np.bincount(line)))
                #             utils.confusion_plot(np.array(preds),data.y_test)
                #这里就是 某个样本 进行5次预测，每次预测产生 10个类别的概率，5次 10个类别概率相加找最大的
                prod_preds = np.argmax(np.sum(prods_5_shot, axis=0), axis=1).reshape(-1)

                score_5_shot = accuracy_score(data.y_test, np.array(preds)) * 100
                print('5_shot:', score_5_shot)

                score_5_shot_prod = accuracy_score(data.y_test, prod_preds) * 100
                print('5_shot_prod:', score_5_shot_prod)

                scores_1_shot.append(scores[0])
                scores_5_shot.append(score_5_shot)
                scores_5_shot_prod.append(score_5_shot_prod)

                # test wdcnn
                score = wdcnn_net.evaluate(X_test_noise, y_test, verbose=0)[1] * 100
                print('wdcnn:', score)
                scores_wdcnn.append(score)

        a = pd.DataFrame(np.array(scores_1_shot).reshape(-1, len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_1_shot.csv" % (exp_name, exp), index=True)

        a = pd.DataFrame(np.array(scores_5_shot).reshape(-1, len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot.csv" % (exp_name, exp), index=True)

        a = pd.DataFrame(np.array(scores_5_shot_prod).reshape(-1, len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_5_shot_prod.csv" % (exp_name, exp), index=True)

        a = pd.DataFrame(np.array(scores_wdcnn).reshape(-1, len(snrs)))
        a.columns = snrs
        a.to_csv("tmp/%s/size_%s/scores_wdcnn.csv" % (exp_name, exp), index=True)


EXPAB_train_and_test(exp_name, exps, is_training)
#
np.bincount([2,2,3,3,1])
#
def EXPAB_analysis(exp_name, exps):
    scores_1_shot_all = pd.DataFrame()
    scores_5_shot_all = pd.DataFrame()
    scores_5_shot_prod_all = pd.DataFrame()
    scores_wdcnn_all = pd.DataFrame()
    for exp in exps:
        file_path = "tmp/%s/size_%s" % (exp_name, exp)
        tmp_data = pd.read_csv("%s/scores_1_shot.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_1_shot_all = pd.concat([scores_1_shot_all, tmp_data], axis=0)

        tmp_data = pd.read_csv("%s/scores_5_shot.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_5_shot_all = pd.concat([scores_5_shot_all, tmp_data], axis=0)

        tmp_data = pd.read_csv("%s/scores_5_shot_prod.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_5_shot_prod_all = pd.concat([scores_5_shot_prod_all, tmp_data], axis=0)

        tmp_data = pd.read_csv("%s/scores_wdcnn.csv" % (file_path),
                               sep=',', index_col=0)
        tmp_data['exp'] = exp
        scores_wdcnn_all = pd.concat([scores_wdcnn_all, tmp_data], axis=0)

    scores_1_shot_all.to_csv("tmp/%s/scores_1_shot_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_5_shot_all.to_csv("tmp/%s/scores_5_shot_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_5_shot_prod_all.to_csv("tmp/%s/scores_5_shot_prob_all.csv" % (exp_name), float_format='%.6f', index=True)
    scores_wdcnn_all.to_csv("tmp/%s/scores_wdcnn_all.csv" % (exp_name), float_format='%.6f', index=True)

    scores_1_shot_all['model'] = 'One-shot'
    scores_5_shot_all['model'] = 'Five-shot'
    scores_5_shot_prod_all['model'] = 'Five-shot-prob'
    scores_wdcnn_all['model'] = 'WDCNN'

    scores_all = pd.concat([scores_1_shot_all, scores_5_shot_all, scores_5_shot_prod_all, scores_wdcnn_all], axis=0)
    scores_all.to_csv("tmp/%s/scores_all.csv" % (exp_name), float_format='%.6f', index=True)

    return scores_all

#
# analysis
scores_all = EXPAB_analysis(exp_name,exps)
scores_all_mean = scores_all.groupby(['model','exp']).mean()
scores_all_std = scores_all.groupby(['model','exp']).std()
scores_all_mean.to_csv("tmp/%s/scores_all_mean.csv" % (exp_name), float_format='%.2f', index=True)
scores_all_std.to_csv("tmp/%s/scores_all_std.csv" % (exp_name), float_format='%.2f', index=True)
scores_all_mean, scores_all_std

#
from sklearn.metrics import accuracy_score
import keras

num = 90
train_classes = sorted(list(set(data.y_train)))
train_indices = [np.where(data.y_train == i)[0] for i in train_classes]

train_idxs = []
val_idxs = []
for i, c in enumerate(train_classes):
    select_idx = train_indices[i][np.random.choice(len(train_indices[i]), num, replace=False)]
    split = int(0.6 * num)
    train_idxs.extend(select_idx[:split])
    val_idxs.extend(select_idx[split:])
X_train, y_train = data.X_train[train_idxs], data.y_train[train_idxs]
X_val, y_val = data.X_train[val_idxs], data.y_train[val_idxs]

siamese_loader = siamese.Siamese_Loader(X_train,
                                        y_train,
                                        data.X_test,
                                        data.y_test)

siamese_net = models.load_siamese_net()
wdcnn_net = models.load_wdcnn_net()

settings["save_path"] = "tmp/%s/size_%s/time_%s/" % (exp_name, num, 0)
siamese_net.load_weights(settings["save_path"] + settings['save_weights_file'])
wdcnn_net.load_weights("%s/weights-best-10-cnn-low-data.hdf5" % (settings["save_path"]))

y_test = keras.utils.to_categorical(data.y_test, data.nclasses)

#
from keras import backend as K
import numpy as np
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')

intermediate_tensor_function = K.function([siamese_net.layers[2].layers[0].input],
                                          [siamese_net.layers[2].layers[-1].output])

plot_only = len(data.y_test)
intermediate_tensor = intermediate_tensor_function([data.X_test[0:plot_only]])[0]
# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(intermediate_tensor)
p_data = pd.DataFrame(columns=['x', 'y', 'label'])
p_data.x = low_dim_embs[:, 0]
p_data.y = low_dim_embs[:, 1]
p_data.label = data.y_test[0:plot_only]
utils.plot_with_labels(p_data)
plt.savefig("%s/90-tsne-one-shot.pdf" % (settings["save_path"]))
#
from keras import backend as K
import numpy as np

intermediate_tensor_function = K.function([wdcnn_net.layers[1].layers[0].input],
                                          [wdcnn_net.layers[1].layers[-1].output])
plot_only = len(data.y_test)
intermediate_tensor = intermediate_tensor_function([data.X_test[0:plot_only]])[0]
# Visualization of trained flatten layer (T-SNE)
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(intermediate_tensor)
import pandas as pd
p_data = pd.DataFrame(columns=['x', 'y', 'label'])
p_data.x = low_dim_embs[:, 0]
p_data.y = low_dim_embs[:, 1]
p_data.label = data.y_test[0:plot_only]
utils.plot_with_labels(p_data)
plt.savefig("%s/90-tsne-wdcnn.pdf" % (settings["save_path"]))
#
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
s = 'val'
val_acc,preds, probs_all = siamese_loader.test_oneshot2(siamese_net,len(siamese_loader.classes[s]),len(siamese_loader.data[s]),verbose=False)
# utils.confusion_plot(preds[:,1],preds[:,0])
utils.plot_confusion_matrix(confusion_matrix(data.y_test,preds[:,1]),  normalize=False,  title=None)
plt.savefig("%s/90-cm-one-shot.pdf" % (settings["save_path"]))
#
pred = np.argmax(wdcnn_net.predict(data.X_test), axis=1).reshape(-1,1)
# utils.confusion_plot(pred,data.y_test)
utils.plot_confusion_matrix(confusion_matrix(data.y_test,pred),  normalize=False,title=None)
plt.savefig("%s/90-cm-wdcnn.pdf" % (settings["save_path"]))