# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D, Conv1D, Lambda, merge, Dense, Flatten, MaxPooling2D, MaxPooling1D, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam,RMSprop
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import time
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers
import keras

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def load_siamese_net(input_shape=(2048, 2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    convnet.add(
        Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100, activation='sigmoid'))

    #     print('WDCNN convnet summary:')
    #     convnet.summary()

    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1, activation='sigmoid')(D1_layer)

    # #下面是自己修改的输出
    # prediction=Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])
    #

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    optimizer = Adam(0.00006)
    #optimizer = RMSprop()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    # # 修改损失函数
    # siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())

    return siamese_net

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
from keras import regularizers

def load_siamese_net_my_weiluansheng(input_shape=(2048, 2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    convnet.add(
        Conv1D(filters=16, kernel_size=64,
                kernel_regularizer=regularizers.l2(0.01),strides=16, activation='relu', padding='same', input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100, activation='sigmoid'))

    #     print('WDCNN convnet summary:')
    #     convnet.summary()
    convnet2 = Sequential()

    # WDCNN
    convnet2.add(
        Conv1D(filters=16, kernel_size=64, strides=16,kernel_regularizer=regularizers.l2(0.01), activation='relu', padding='same', input_shape=input_shape))
    convnet2.add(MaxPooling1D(strides=2))
    convnet2.add(Conv1D(filters=32, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet2.add(MaxPooling1D(strides=2))
    convnet2.add(Conv1D(filters=64, kernel_size=2, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet2.add(MaxPooling1D(strides=2))
    convnet2.add(Conv1D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu', padding='same'))
    convnet2.add(MaxPooling1D(strides=2))
    convnet2.add(Conv1D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(0.01), strides=1, activation='relu'))
    convnet2.add(MaxPooling1D(strides=2))
    convnet2.add(Flatten())
    convnet2.add(Dense(100, activation='sigmoid'))
    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet2(right_input)

    # # layer to merge two encoded inputs with the l1 distance between them
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # # call this layer on list of two input tensors.
    # L1_distance = L1_layer([encoded_l, encoded_r])
    # D1_layer = Dropout(0.5)(L1_distance)
    # prediction = Dense(1, activation='sigmoid')(D1_layer)

    # #下面是自己修改的输出
    prediction = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # optimizer = Adam(0.00006)
    optimizer = RMSprop()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

    # # 修改损失函数
    # siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())

    return siamese_net
def res_block_v1(x, input_filter, output_filter):
    res_x = Conv1D(kernel_size=3, filters=output_filter, activation='relu',strides=1, padding='same')(x)
    #res_x = layers.BatchNormalization()(res_x)
    #res_x = layers.Activation('relu')(res_x)
    # res_x = Conv2D(kernel_size=3, filters=output_filter, strides=1, padding='same')(res_x)
    # res_x = layers.BatchNormalization()(res_x)
    if input_filter == output_filter:
        identity = x
    else: #需要升维或者降维了就是对原先的输入进行1*1的卷积，目前看代码就是卷积核数目不一样就是通道不一样啊
        identity = Conv1D(kernel_size=1, filters=output_filter, strides=1, padding='same')(x)
    x = keras.layers.add([identity, res_x])
    output = layers.Activation('relu')(x)
    return output
def load_siamese_net_my_mew(input_shape=(2048, 2)):
    inputs = layers.Input(input_shape)


   #Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))

    net = layers.Conv1D(filters=16, kernel_regularizer=l2(1e-4),kernel_size=64,strides=16,padding='same',activation = 'relu')(inputs)
    net=MaxPooling1D(strides=2)(net)
    #net=Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(net)
    net=res_block_v1(net,16,32)
    net=MaxPooling1D(strides=2)(net)
    #net=Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same')(net)
    net = res_block_v1(net, 32, 64)
    net=MaxPooling1D(strides=2)(net)
    #net=Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(net)
    net = res_block_v1(net, 64, 64)
    net =MaxPooling1D(strides=2)(net)
    #net=Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')(net)
    net = res_block_v1(net, 64, 64)
    net=MaxPooling1D(strides=2)(net)
    net =Flatten()(net)
    net=Dense(100, activation='sigmoid')(net)
    model = Model(inputs=inputs, outputs=net)

    #     print('WDCNN convnet summary:')
    #     convnet.summary()
    left_input = layers.Input(input_shape)
    right_input = layers.Input(input_shape)
    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = model(left_input)
    encoded_r = model(right_input)


    # # layer to merge two encoded inputs with the l1 distance between them
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # # call this layer on list of two input tensors.
    # L1_distance = L1_layer([encoded_l, encoded_r])
    # D1_layer = Dropout(0.5)(L1_distance)
    # prediction = Dense(1, activation='sigmoid')(D1_layer)

    # #下面是自己修改的输出
    prediction=Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    #optimizer = Adam(0.00006)
    optimizer = RMSprop()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss=contrastive_loss, optimizer=optimizer,metrics=[accuracy])

    # # 修改损失函数
    # siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())

    return siamese_net

def load_siamese_net_my(input_shape=(2048, 2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    convnet.add(
        Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100, activation='sigmoid'))

    #     print('WDCNN convnet summary:')
    #     convnet.summary()

    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)


    # # layer to merge two encoded inputs with the l1 distance between them
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # # call this layer on list of two input tensors.
    # L1_distance = L1_layer([encoded_l, encoded_r])
    # D1_layer = Dropout(0.5)(L1_distance)
    # prediction = Dense(1, activation='sigmoid')(D1_layer)

    # #下面是自己修改的输出
    prediction=Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    #optimizer = Adam(0.00006)
    optimizer = RMSprop()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss=contrastive_loss, optimizer=optimizer,metrics=[accuracy])

    # # 修改损失函数
    # siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())

    return siamese_net

def load_siamese_net_2channel(input_shape=(4096, 2)):
    left_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    convnet.add(
        Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100, activation='sigmoid'))

    #     print('WDCNN convnet summary:')
    #     convnet.summary()

    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)


    prediction = Dense(1, activation='sigmoid')(encoded_l)
    siamese_net = Model(inputs=left_input, outputs=prediction)

    # optimizer = Adam(0.00006)
    optimizer = Adam()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    #siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
    #修改损失函数
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)


    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())

    return siamese_net

def load_wdcnn_net(input_shape=(2048, 2), nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()

    # WDCNN
    convnet.add(
        Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100, activation='sigmoid'))

    #     print('convnet summary:')
    # convnet.summary()

    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(nclasses, activation='softmax')(Dropout(0.5)(encoded_cnn))
    wdcnn_net = Model(inputs=left_input, outputs=prediction_cnn)

    # optimizer = Adam(0.00006)
    optimizer = Adam()
    wdcnn_net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print('\nsiamese_net summary:')
    # cnn_net.summary()
    print(wdcnn_net.count_params())
    return wdcnn_net



def abs_backend(inputs):
    return K.abs(inputs)

    # 在索引“ axis”处添加1个大小的尺寸。


def expand_dim_backend(inputs):
    return K.expand_dims(inputs, 1)

    # def expand_dim_backend(inputs):
    #     return K.expand_dims(K.expand_dims(inputs,1),1)


def sign_backend(inputs):
    return K.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = K.expand_dims(inputs, -1)
    inputs = K.spatial_2d_padding(inputs, ((0, 0), (pad_dim, pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

    # Residual Shrinakge Block


def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2, kernel_size=3):
    '''

    :param incoming: 输入
    :param nb_blocks: 有几个RSBU单元
    :param out_channels:  输出通道数，或者说是上一层输出的通道数
    :param downsample: 是否降采样 如果不是downsample_strides=1 否则就是默认值2
    :param downsample_strides:
    :param kernel_size: 指定卷积核大小，默认是3
    :return:
    '''
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    for i in range(nb_blocks):

        identity = residual

        if not downsample:
            downsample_strides = 1

        residual = layers.BatchNormalization()(residual)
        residual = layers.Activation('relu')(residual)
        # print(residual.shape) #(?, 1, 64, 16) 不对，正确是(?, 128, 16)
        residual = layers.Conv1D(filters=out_channels,
                                 kernel_size=kernel_size,
                                 strides=downsample_strides,
                                 padding='same',
                                 kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(residual)

        residual = layers.BatchNormalization()(residual)
        residual = layers.Activation('relu')(residual)
        # 512 16
        residual = layers.Conv1D(out_channels, kernel_size, padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(1e-4))(residual)

        # Calculate global means
        residual_abs = layers.Lambda(abs_backend)(residual)
        abs_mean = layers.GlobalAveragePooling1D()(residual_abs)

        # Calculate scaling coefficients
        scales = layers.Dense(out_channels, activation=None, kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-4))(abs_mean)
        scales = layers.BatchNormalization()(scales)
        scales = layers.Activation('relu')(scales)
        scales = layers.Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = layers.Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([layers.Lambda(sign_backend)(residual), n_sub])

        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = layers.AveragePooling1D(pool_size=1, strides=2)(identity)  # 1024 16 -》 512 16

        # Zero_padding to match channels
        if in_channels != out_channels:  # 第一次执行时？ 256 16   16变32通道
            identity = layers.Lambda(pad_backend,
                                     arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)

        residual = keras.layers.add([residual, identity])

    return residual


def load_siamese_net_drsn(input_shape=(2048, 2), nclasses=10):
    inputs = Input(input_shape)
    net = layers.Conv1D(filters=16,
                        kernel_size=64,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(inputs)
    net = residual_shrinkage_block(net, 1, 16, downsample=True, kernel_size=32)
    net = residual_shrinkage_block(net, 1, 16, downsample=False, kernel_size=32)

    net = residual_shrinkage_block(net, 1, 32, downsample=True, kernel_size=16)
    net = residual_shrinkage_block(net, 1, 32, downsample=False, kernel_size=16)

    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    # net = layers.BatchNormalization()(net)
    # net = layers.Activation('relu')(net)

    net = layers.GlobalAveragePooling1D()(net)
    net = layers.Dense(100, activation='relu')(net)

    #     print('WDCNN convnet summary:')
    #     convnet.summary()

    # call the convnet Sequential model on each of the input tensors so params will be shared
    model = Model(inputs=inputs, outputs=net)

    left_input = Input(input_shape)
    right_input = Input(input_shape)
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    # layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1, activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # optimizer = Adam(0.00006)
    optimizer = Adam()
    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())
    return siamese_net


def load_drsn_net(input_shape=(2048, 2), nclasses=10):
    inputs = layers.Input(input_shape)
    # 1024 16
    net = layers.Conv1D(filters=16,
                        kernel_size=64,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(inputs)
    net = residual_shrinkage_block(net, 1, 16, downsample=True, kernel_size=32)
    net = residual_shrinkage_block(net, 1, 16, downsample=False, kernel_size=32)

    net = residual_shrinkage_block(net, 1, 32, downsample=True, kernel_size=16)
    net = residual_shrinkage_block(net, 1, 32, downsample=False, kernel_size=16)

    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=True, kernel_size=8)
    net = residual_shrinkage_block(net, 1, 64, downsample=False, kernel_size=8)
    # net = layers.BatchNormalization()(net)
    # net = layers.Activation('relu')(net)

    net = layers.GlobalAveragePooling1D()(net)
    outputs = layers.Dense(100, activation='relu')(net)

    outputs = layers.Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    siamese_net = load_siamese_net_my_mew()
    print('\nsiamese_net summary:')
    from keras.utils import plot_model

    #plot_model(siamese_net, to_file='model.png', show_shapes=True)
    siamese_net.summary()
    from keras.utils.vis_utils import plot_model

    #plot_model(siamese_net, to_file='model2.png', show_shapes=True)
    print('\nsequential_3 is WDCNN:')
    siamese_net.layers[2].summary()
    load_drsn_net()