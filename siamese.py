import numpy.random as rng
from sklearn.utils import shuffle
import numpy as np
import seaborn
import os, json

from sys import stdout


def flush(string):
    stdout.write('\r')
    stdout.write(str(string))
    #手动刷新缓冲区 显示输出
    stdout.flush()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, X_train, y_train, X_val, y_val):
        self.data = {'train': X_train, 'val': X_val}
        print("Siamese_Loader的X_train的shape",X_train.shape)
        print("Siamese_Loader的y_train的shape",y_train.shape)
        print("Siamese_Loader的X_val的shape",X_val.shape)
        print("Siamese_Loader的y_val的shape",y_val.shape)

        self.labels = {'train': y_train, 'val': y_val}
        train_classes = list(set(y_train))
        np.random.seed(10)
        #         train_classes = sorted(rng.choice(train_classes,size=(int(len(train_classes)*0.8),),replace=False) )
        self.classes = {'train': sorted(train_classes), 'val': sorted(list(set(y_val)))}
        self.indices = {'train': [np.where(y_train == i)[0] for i in self.classes['train']],
                        'val': [np.where(y_val == i)[0] for i in self.classes['val']]
                        }
        #print("Siamese_Loader类的类别数:", self.classes)
        #print("Siamese_Loader类X_train数:", len(X_train), "Siamese_Loader类X_val数:", len(X_val))
        #print([len(c) for c in self.indices['train']], [len(c) for c in self.indices['val']])

    def set_val(self, X_val, y_val):
        self.data['val'] = X_val
        self.labels['val'] = y_val
        self.classes['val'] = sorted(list(set(y_val)))
        self.indices['val'] = [np.where(y_val == i)[0] for i in self.classes['val']]

    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w, h = X.shape

        # randomly sample several classes to use in the batch
        #随机抽取几个类用于批处理
        categories = rng.choice(n_classes, size=(batch_size,), replace=True)

        # initialize 2 empty arrays for the input image batch
        # 为输入图像批初始化2个空数组
        pairs = [np.zeros((batch_size, w, h, 1)) for i in range(2)]

        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        #初始化目标的向量，并将其一半设为“1”，这样批处理的下半部分就具有相同的类
        targets = np.zeros((batch_size,))
        ####尝试修改不一样类别从0 变为-1
        # targets = np.full([batch_size,], -1)
        ####

        targets[batch_size // 2:] = 1  # 后batch_size/2  表示同类的1 不同类的是0
        for i in range(batch_size):
            category = categories[i]
            n_examples = len(X_indices[category])  #category该类对应的样本下标 长度
            if (n_examples == 0):
                print("error:n_examples==0", n_examples)
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[X_indices[category][idx_1]].reshape(w, h, 1)
            # pick images of same class for 1st half, different for 2nd
            ##为上半场挑选相同级别的图片，为下半场挑选不同级别的图片
            if i >= batch_size // 2:
                category_2 = category
                idx_2 = (idx_1 + rng.randint(1, n_examples)) % n_examples
            else:
                # add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
                n_examples = len(X_indices[category_2])
                idx_2 = rng.randint(0, n_examples)
            pairs[1][i, :, :, :] = X[X_indices[category_2][idx_2]].reshape(w, h, 1)
        return pairs, targets, categories

    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)

    def make_oneshot_task(self, N, s="val", language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X = self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w, h = X.shape
        if N > n_classes:
            raise ValueError("{} way task has greter than {} classes".format(N, n_classes))

        categories = rng.choice(n_classes, size=(N,), replace=False)
        true_category = categories[0]
        n_examples = len(X_indices[true_category])
        ex1, ex2 = rng.choice(n_examples, size=(2,), replace=False)
        test_image = np.asarray([X[X_indices[true_category][ex1]]] * N).reshape(N, w, h, 1)
        support_set = np.zeros((N, w, h))
        support_set[0, :, :] = X[X_indices[true_category][ex2]]
        for idx, category in enumerate(categories[1:]):
            n_examples = len(X_indices[category])
            support_set[idx + 1, :, :] = X[X_indices[category][rng.randint(0, n_examples)]]
        support_set = support_set.reshape(N, w, h, 1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set, categories = shuffle(targets, test_image, support_set, categories)
        pairs = [test_image, support_set]

        return pairs, targets, categories

    def test_oneshot(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        val_c = self.labels[s]
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k, N))
        preds = []
        err_print_num = 0
        for idx in range(k):
            inputs, targets, categories = self.make_oneshot_task(N, s)
            n_classes, w, h, _ = inputs[0].shape
            #             inputs[0]=inputs[0].reshape(n_classes,100,100,h)
            #             inputs[1]=inputs[1].reshape(n_classes,100,100,h)
            inputs[0] = inputs[0].reshape(n_classes, w, h)
            inputs[1] = inputs[1].reshape(n_classes, w, h)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
            elif verbose and err_print_num < 1:
                err_print_num = err_print_num + 1
                print(targets)
                #                 print(categories)
                print([categories[np.argmax(targets)], categories[np.argmax(probs)]])
                inputs[0] = inputs[0].reshape(n_classes, w, h, 1)
                inputs[1] = inputs[1].reshape(n_classes, w, h, 1)
                plot_pairs(inputs, [np.argmax(targets), np.argmax(probs)])
            preds.append([categories[np.argmax(targets)], categories[np.argmax(probs)]])
        #             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
        return percent_correct, preds

    def make_oneshot_task2(self, idx, s="val"):
        """Create pairs_list of test image, support set for testing N way one-shot learning.
        创建测试图像的列表，为测试N路单次学习设置支持
        """
        X = self.data[s]
        X_labels = self.labels[s]

        X_train = self.data['train']
        indices_train = self.indices['train']
        classes_train = self.classes['train']
        N = len(indices_train)

        _, w, h = X.shape
        # X[idx] 复制了十份 然后reshape后，shape:10 2048 2 1
        test_image = np.asarray([X[idx]] * N).reshape(N, w, h, 1)
        support_set = np.zeros((N, w, h))
        for index in range(N):
            # 从 N个类别里面 中的index里面 的所有样本中 随机选择一个样本的下标记
            support_set[index, :, :] = X_train[rng.choice(indices_train[index], size=(1,), replace=False)]
        support_set = support_set.reshape(N, w, h, 1)  # shape:10 2048 2 1

        targets = np.zeros((N,))
        ## 修改 修改0 到-1
        # targets = np.full([N,], -1)
        ###

        true_index = classes_train.index(X_labels[idx])  #训练样本的真是类别
        targets[true_index] = 1

        # targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, classes_train)
        categories = classes_train

        pairs = [test_image, support_set]

        return pairs, targets, categories

    def test_oneshot2(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        #labels存储的是多个样本的类别标签  K表示labels[s]的长度
        k = len(self.labels[s])
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k, N))
        preds = []
        probs_all = []
        err_print_num = 0
        for idx in range(k):
            # 对于K个样本，制造oneshot，其实就是将这个待验证样本重复 10次（10个类别) 然后
            # 从其他的10个类别中挑选出 一个样本与 这个待验证样本 组合
            #inputs 是一个list 含有两个ndarra ，每个ndarray shape为  10 2048 2  1
            inputs, targets, categories = self.make_oneshot_task2(idx, s)
            n_classes, w, h, _ = inputs[0].shape
            inputs[0] = inputs[0].reshape(n_classes, w, h)
            inputs[1] = inputs[1].reshape(n_classes, w, h)
            # 长度为10的数组，表示每个类别的概率
            probs = model.predict(inputs)
            if np.argmin(probs) == np.argmax(targets):  #修改的
                n_correct += 1
            elif verbose and err_print_num < 1:
                err_print_num = err_print_num + 1
                print(targets)
                #                 print(categories)
                print([categories[np.argmax(targets)], categories[np.argmax(probs)]])
                inputs[0] = inputs[0].reshape(n_classes, w, h, 1)
                inputs[1] = inputs[1].reshape(n_classes, w, h, 1)
                plot_pairs(inputs, [np.argmax(targets), np.argmax(probs)])
            # [[2,4],[3,3] 表示每次预测类别和正确类别  修改的
            preds.append([categories[np.argmax(targets)], categories[np.argmin(probs)]])
            # 保存所有的预测结果 每一次预测结果是一个ndarray shape为10
            probs_all.append(probs)
        #             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
        return percent_correct, np.array(preds), np.array(probs_all)

    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size), )


def train_and_test_oneshot(settings, siamese_net, siamese_loader):
    """
    进行孪生网络的训练
    :param settings:
    :param siamese_net:
    :param siamese_loader:
    :return:
    """
    settings['best'] = -1
    settings['n'] = 0
    # {'N_way': 10, 'batch_size': 32, 'best': -1, 'evaluate_every': 300, 'loss_every': 20,
    # 'n_iter': 15000, 'n_val': 2, 'n': 0, 'save_path': 'tmp/EXP-C/size_90/0.7/time_0/'
    # , 'save_weights_file': 'weights-best-10-oneshot-low-data.hdf5'}

    print(settings)

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print("training...")

    # Training loop
    for i in range(settings['n'], settings['n_iter']):  # 0-15000
        #inputs:得到2  ndarray 的batch 组成的list，targets：barch_size个0 1，前一半是0 后一半是1
        (inputs, targets, _) = siamese_loader.get_batch(settings['batch_size'])
        #一个batch的shape    batch_size ,2048 ,2
        # 这里n_classes 表示的是 barch_size
        n_classes, w, h, _ = inputs[0].shape

        # 从32 2048 2 1 变为32 2048 2
        inputs[0] = inputs[0].reshape(n_classes, w, h)
        inputs[1] = inputs[1].reshape(n_classes, w, h)

        #  inputs list中两个ndarray 表示两个输入
        # https://zhuanlan.zhihu.com/p/198982185  训练
        loss = siamese_net.train_on_batch(inputs, targets)
        gettt=siamese_net.predict(inputs)
        # if type(loss) is list:
        #     loss=loss[0] # 修改的 ，因为自己设计的对比损失 指定了 metrics 所以会返回损失和 metrics两个值
        # #保存模型
        if i % settings['evaluate_every'] == 0:
            val_acc, preds, probs_all = siamese_loader.test_oneshot2(siamese_net, settings['N_way'], settings['n_val'],
                                                                     verbose=False)
            preds = np.array(preds)
            if val_acc >= settings['best']:
                print("\niteration {} evaluating: {}".format(i, val_acc))
                #                 print(loader.classes)
                #                 score(preds[:,1],preds[:,0])
                #                 print("\nsaving")
                siamese_net.save(weights_path)
                settings['best'] = val_acc
                settings['n'] = i
                with open(os.path.join(weights_path + ".json"), 'w') as f:
                    f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': ')))

        if i % settings['loss_every'] == 0:
            val_acc, preds, probs_all = siamese_loader.test_oneshot2(siamese_net, settings['N_way'], settings['n_val'],
                                                                     verbose=False)
            if type(loss) is list:
                flush("{} 轮: 损失值：{:.5f}, 正确率:{:.5f}，验证集正确率{:.5f}:".format(i, loss[0], loss[1], val_acc))
            else:
                flush("{} : {:.5f},".format(i, loss))

    return settings['best']