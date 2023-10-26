# from __future__ import division
import argparse
import csv
import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from dataset.tf_processdata import train_input_fn, eval_input_fn
from model import Model, MyModel
from utils.myCallback import HistoryRecord, P_MRR_Metric
from utils.recorder import Recorder
from utils.loss import Loss_with_L2


def extract(x, y):
    return y


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()

MODE = 'new'

print('Loading train data.')
assert opt.dataset

n_node = 0
max_seq = 0
max_n_node = 0

with open(f"dataset/{opt.dataset}/train.csv", "r") as data_file:
    data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]
    n_node = max(n_node, np.amax([np.amax(z) for z in data]) + 1)
    max_seq = max(max_seq, len(max(data, key=len)))
    max_n_node = max(max_n_node, len(max([np.unique(i) for i in data], key=len)))
    train_dataset_size = len(data)

with open(f"dataset/{opt.dataset}/test.csv", "r") as data_file:
    data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]
    n_node = max(n_node, np.amax([np.amax(z) for z in data]) + 1)
    max_seq = max(max_seq, len(max(data, key=len)))
    max_n_node = max(max_n_node, len(max([np.unique(i) for i in data], key=len)))
    test_dataset_size = len(data)

print("=== Train dataset ===")
print("Dataset size:", train_dataset_size)

print("=== Test dataset ===")
print("Dataset size:", test_dataset_size)

print("=== Both datasets ===")
print("Longest sequence:", max_seq)
print("Total unique items:", n_node)
print("Highest number of unique items in a session:", max_n_node)

print('Processing train data.')
train_data = train_input_fn(opt.batchSize, max_seq, max_n_node)

print('Processing test data.')
test_data = eval_input_fn(opt.batchSize, max_seq, max_n_node)

print('Loading model.')

if MODE == 'old':
    model = Model(n_node=n_node,
                  l2=opt.l2,
                  step=opt.step,
                  lr=opt.lr,
                  decay=opt.lr_dc_step * train_dataset_size / opt.batchSize,
                  lr_dc=opt.lr_dc,
                  hidden_size=opt.hiddenSize,
                  out_size=opt.hiddenSize)

    best_result = [0, 0]
    best_epoch = [0, 0]

    recorder = Recorder(1, os.path.join("logs/log", str(int(datetime.datetime.now().timestamp()))))

    for epoch in range(opt.epoch):
        print('start training: ', datetime.datetime.now())

        loss_ = []
        with tqdm(total=np.floor(train_dataset_size / opt.batchSize) + 1) as pbar:
            for A_in, A_out, alias_inputs, items, mask, labels in train_data:
                pbar.update(1)
                train_loss, logits = model.train_step(item=items, adj_in=A_in, adj_out=A_out, mask=mask,
                                                      alias=alias_inputs,
                                                      labels=labels)
                train_loss = train_loss.numpy()
                pbar.set_description(f"Training model. Epoch: {epoch}")
                pbar.set_postfix(loss=train_loss)

                loss_.append(train_loss)

        hit, mrr, test_loss_ = [], [], []
        with tqdm(total=np.floor(test_dataset_size / opt.batchSize) + 1) as pbar:
            for A_in, A_out, alias_inputs, items, mask, labels in test_data:
                pbar.update(1)
                test_loss, logits = model.train_step(item=items, adj_in=A_in, adj_out=A_out, mask=mask,
                                                     alias=alias_inputs,
                                                     labels=labels, train=False)
                test_loss = test_loss.numpy()
                pbar.set_description(f"Testing model. Epoch: {epoch}")
                pbar.set_postfix(loss=test_loss)
                test_loss_.append(test_loss)

                index = np.argsort(logits, 1)[:, -20:]

                for score, target in zip(index, labels):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))

        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        test_loss = np.mean(test_loss_)
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
        recorder.epoch_end(epoch, train_loss, test_loss, best_result[0], best_result[1])

        print(
            f"train_loss: {train_loss}, test_loss: {test_loss}, Recall@20: {best_result[0]}, MMR@20: {best_result[1]}, "
            f"Best epoch: {best_epoch[0]}:{best_epoch[1]}")
elif MODE == 'new':
    save_dir = "logs"
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    # new
    model = MyModel(hidden_size=100, out_size=100, batch_size=100, n_node=n_node)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                              decay_rate=0.1,
                                                              decay_steps=opt.lr_dc_step * train_dataset_size / opt.batchSize,
                                                              staircase=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                                   min_delta=0,
                                                   patience=5,
                                                   verbose=1,
                                                   mode="max")
    checkpoint_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, "best_weights.h5"),
                                                      monitor='MRR@20',
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      save_freq='epoch')
    history_recorder = HistoryRecord(log_dir=os.path.join(save_dir, 'log_' + time_str))
    p_mrr = P_MRR_Metric(val_data=test_data, performance_mode=1, val_size=int(test_dataset_size / opt.batchSize))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  # loss=keras.losses.SparseCategoricalCrossentropy(),
                  loss=Loss_with_L2(model=model, name='scc_loss_with_l2'),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  run_eagerly=False,
                  steps_per_execution=1)
    h = model.fit(x=train_data,
                  epochs=30,
                  verbose=1,
                  callbacks=[p_mrr, history_recorder, checkpoint_best, early_stopping],
                  validation_data=test_data
                  )
    # print(model.trainable_variables)
    # predict_result = model.predict(x=test_data,
    #                                verbose=1)
    # print("y的形状是{}".format(predict_result.shape))
    # print("y的形状是{}".format(predict_result.shape[0]))
else:
    print("other test")
    y = test_data.map(extract)
    print(int(test_dataset_size / opt.batchSize))
