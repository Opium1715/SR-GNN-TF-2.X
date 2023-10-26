import os

import scipy.signal
from matplotlib import pyplot as plt


class Recorder:
    def __init__(self, interval, log_dir):
        self.interval = interval
        self.log_dir = log_dir
        self.train_losses = []
        self.test_losses = []
        self.P = []
        self.MRR = []

    def preset(self, train_losses, test_losses, P, MMR):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.P = P
        self.MRR = MMR

    def epoch_end(self, epoch, train_loss, test_loss, P, MRR):
        if epoch % self.interval == 0:
            pass
        else:
            return
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.MRR.append(MRR)
        self.P.append(P)
        with open(f"{self.log_dir}/epoch_loss.txt", 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_test_loss.txt", 'a') as f:
            f.write(str(test_loss))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_P.txt", 'a') as f:
            f.write(str(P))
            f.write("\n")
        with open(f"{self.log_dir}/epoch_MRR.txt", 'a') as f:
            f.write(str(MRR))
            f.write("\n")
        self._loss_plot()
        self._P_MRR_plot()

    def _P_MRR_plot(self):
        iters = range(len(self.P))
        plt.figure()
        plt.plot(iters, self.P, 'blue', linewidth=2, label='P@20')
        plt.plot(iters, self.MRR, 'coral', linewidth=2, label='MRR@20')

        try:
            if len(self.P) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.P, num, 3), 'green', linestyle='--',
                     linewidth=2, label='smooth P@20')
            plt.plot(iters, scipy.signal.savgol_filter(self.MRR, num, 3), '#8B4513', linestyle='--',
                     linewidth=2, label='smooth MMR@20')
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
        iters = range(len(self.train_losses))
        plt.figure()
        plt.plot(iters, self.train_losses, 'blue', linewidth=2, label='train_loss')
        plt.plot(iters, self.test_losses, 'coral', linewidth=2, label='val_loss')

        try:
            if len(self.train_losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.train_losses, num, 3), 'green', linestyle='--',
                     linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.test_losses, num, 3), '#8B4513', linestyle='--',
                     linewidth=2, label='smooth val loss')
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


if __name__ == '__main__':
    train_loss = []
    val_loss = []
    MRR = []
    P = []
    recoder = Recorder(1, "logs")
    with open("../logs/log_20231014_1/epoch_loss.txt", 'r') as f:
        for line in f.readlines():
            train_loss.append(float(line.strip()))
    with open("../logs/log_20231014_1/epoch_test_loss.txt", 'r') as f:
        for line in f.readlines():
            val_loss.append(float(line.strip()))
    with open("../logs/log_20231014_1/epoch_MRR.txt", 'r') as f:
        for line in f.readlines():
            MRR.append(float(line.strip()))
    with open("../logs/log_20231014_1/epoch_P.txt", 'r') as f:
        for line in f.readlines():
            P.append(float(line.strip()))
    recoder.preset(train_loss,val_loss,P,MRR)
    recoder._loss_plot()
    recoder._P_MRR_plot()

    # recoder.epoch_end(5, 1.22, 1.56, 22.0, 45.6)
