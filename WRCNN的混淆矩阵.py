
from sklearn.metrics import accuracy_score
import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import imp
import pandas as pd

#
import cwru
import siamese
import  models
import  utils
window_size=2048

num = 60

data = cwru.CWRU(['12DriveEndFault'], ['1772', '1750', '1730'], window_size)

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

siamese_net = models.load_siamese_net_my_mew()
#wdcnn_net = models.load_wdcnn_net()

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
exp_name = "EXP-AB发论文"

settings["save_path"] = "tmp/%s/size_%s/time_%s/" % (exp_name, num, 0)
siamese_net.load_weights(settings["save_path"] + settings['save_weights_file'])
#wdcnn_net.load_weights("%s/weights-best-10-cnn-low-data.hdf5" % (settings["save_path"]))

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



#
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
s = 'val'
val_acc,preds, probs_all = siamese_loader.test_oneshot2(siamese_net,len(siamese_loader.classes[s]),len(siamese_loader.data[s]),verbose=False)
# utils.confusion_plot(preds[:,1],preds[:,0])
utils.plot_confusion_matrix(confusion_matrix(data.y_test,preds[:,1]),  normalize=False,  title=None)
plt.savefig("%s/60-cm-one-shot.pdf" % (settings["save_path"]))