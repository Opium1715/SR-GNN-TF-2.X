import argparse
import csv
import datetime
import os

import numpy as np
from tensorflow import keras

from dataset.tf_processdata import train_input_fn, eval_input_fn
from model import MyModel
from utils.loss import Loss_with_L2
from utils.myCallback import HistoryRecord, P_MRR_Metric


def extract(x, y):
    return y


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchSize', type=int, default=100, help='bat')
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
if MODE == 'new':
    save_dir = "logs"
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    # new
    model = MyModel(hidden_size=100, out_size=100, batch_size=100, n_node=n_node)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                              decay_rate=opt.lr_dc,
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
                  loss=Loss_with_L2(model=model, l2=opt.l2, name='scc_loss_with_l2'),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  run_eagerly=True,
                  steps_per_execution=1)
    h = model.fit(x=train_data,
                  epochs=opt.epoch,
                  verbose=1,
                  callbacks=[p_mrr, history_recorder, checkpoint_best, early_stopping],
                  validation_data=test_data
                  )
