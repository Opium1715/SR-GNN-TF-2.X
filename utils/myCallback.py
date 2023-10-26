import numpy as np
import scipy
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm


class HistoryRecord(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.losses = []
        self.val_losses = []
        self.mrr = []
        self.precision = []
        self.accuracy = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('sparse_categorical_accuracy'))
        self.mrr.append(logs.get('MRR@20'))
        self.precision.append(logs.get('P@20'))

        with open(f"{self.log_dir}/epoch_loss.txt", 'a') as f:
            f.write(str(self.losses[-1]))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_test_loss.txt", 'a') as f:
            f.write(str(self.val_losses[-1]))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_P.txt", 'a') as f:
            f.write(str(self.precision[-1]))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_MRR.txt", 'a') as f:
            f.write(str(self.mrr[-1]))
            f.write("\n")
        self._loss_plot()
        self._P_MRR_plot()

    def _P_MRR_plot(self):
        iters = range(len(self.precision))
        plt.figure()
        plt.plot(iters, self.precision, 'blue', linewidth=2, label='P@20')
        plt.plot(iters, self.mrr, 'red', linewidth=2, label='MRR@20')

        try:
            if len(self.precision) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.precision, num, 3), 'cyan', linestyle=':',
                     linewidth=4, label='smooth P@20')
            plt.plot(iters, scipy.signal.savgol_filter(self.mrr, num, 3), 'm', linestyle=':',
                     linewidth=4, label='smooth MMR@20')
        except:
            pass

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('probabilities')
        plt.title('P@20 & MMR@20 Curve')
        plt.legend(loc='upper right')
        plt.savefig(f"{self.log_dir}/epoch_P_MRR.png")
        plt.cla()
        plt.close("all")

    def _loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'blue', linewidth=2, label='train_loss')
        plt.plot(iters, self.val_losses, 'red', linewidth=2, label='val_loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'cyan', linestyle=':',
                     linewidth=4, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), 'm', linestyle=':',
                     linewidth=4, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss Curve')
        plt.legend(loc='upper right')
        plt.savefig(f"{self.log_dir}/epoch_loss.png")
        plt.cla()
        plt.close("all")


def extract_labels(x, y):
    return y


class P_MRR_Metric(tf.keras.callbacks.Callback):
    def __init__(self, val_data, val_size, performance_mode=1):
        super().__init__()
        self.val_data = val_data
        self.performance_mode = performance_mode  # default to batch process
        self.total_val_size = val_size

    def on_epoch_end(self, epoch, logs=None):
        precision = []
        mrr = []
        if self.performance_mode == 0:
            # solution 1：直接一次性全部评估
            predict_result = self.model.predict(x=self.val_data,
                                                verbose=0)
            # predict的shape是[500, node]
            # print("结果的形状是{}".format(predict_result.shape))
            # P@20 MRR@20
            indices = tf.argsort(predict_result, axis=1, direction="DESCENDING")[:, :20]
            # 分离标签
            y_labels = self.val_data.map(extract_labels).batch(batch_size=predict_result.shape[0])
            y = None
            for labels in y_labels:
                y = labels
            y = tf.reshape(tensor=y, shape=[-1, ])
            for index, label in zip(indices, y):
                precision.append(np.isin(label, index))
                if len(np.asarray(index == label).nonzero()[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.asarray(index == label).nonzero()[0][0] + 1))
            precision = np.mean(precision)
            mrr = np.mean(mrr)
            logs['P@20'] = precision
            logs['MRR@20'] = mrr
        # solution 2： 分批次 size=100
        elif self.performance_mode == 1:
            # y_labels = self.val_data.map(extract_labels)
            print('calculate P@20 and MRR@20 of current epoch\n')
            with tqdm(total=self.total_val_size) as processbar:
                for data, labels in self.val_data:
                    predict_result = self.model.predict(x=data,
                                                        batch_size=100,
                                                        verbose=0)
                    indices = tf.argsort(predict_result, axis=1, direction="DESCENDING")[:, :20]
                    for index, label in zip(indices, labels):
                        precision.append(np.isin(label, index))
                        if len(np.asarray(index == label).nonzero()[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1 / (np.asarray(index == label).nonzero()[0][0] + 1))
                    processbar.update(1)
            precision = np.mean(precision)
            mrr = np.mean(mrr)
            logs['P@20'] = precision
            logs['MRR@20'] = mrr
