"""
class to train and test lstm model
"""

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt


class TrainTestLSTM:
    def __init__(self, data, model, learning_rate=0.01,
                 batch_size=128, parallel=False, logger=None):
        self.model = model
        self.batch_size = batch_size
        self.logger = logger
        self.data = data
        self.T = model.T

        if parallel:
            self.model = nn.DataParallel(self.model)

        # set model optimizer
        self.model_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.model.parameters()),
                                            lr=learning_rate)

        # define loss function
        self.loss_func = nn.MSELoss()

    # split and normalize data
    def preprocess_data(self, train_split=0.7):
        # split data
        self.X = self.data[[x for x in self.data.columns if x != "NDX"]][:-1].as_matrix()
        # drop last row since using forward y
        y = self.data.NDX.shift(-1).values
        self.y = y[:-1].reshape((-1, 1))

        # normalize y
        self.train_size = int(self.X.shape[0]*train_split)
        self.y_train_mean = np.mean(self.y[:self.train_size])
        self.y = self.y - self.y_train_mean
        # self.X_train_mean = np.mean(self.X[:self.train_size])
        # self.X = self.X - self.X_train_mean

        if self.logger != None:
            self.logger.info("Training size: %s" %(self.train_size))

    def train(self, n_epochs=10, plot_predict=True):
        # iterations per epoch
        iter_per_epoch = int(np.ceil(self.train_size * 1./self.batch_size))
        if self.logger != None:
            self.logger.info("Iterations per epoch: %s ~ %s" %(self.train_size * 1. / self.batch_size,
                                                               iter_per_epoch))
            self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
            self.epoch_losses = np.zeros(n_epochs)

            n_iter = 0

            for i in range(n_epochs):
                print("\n--------------------------------")
                print("Epoch: ", i)
                perm_idx = np.random.permutation(self.train_size - self.T)
                j=0
                while j < self.train_size - self.T:
                    batch_idx = perm_idx[j: (j+self.batch_size)]
                    X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
                    y_history = np.zeros((len(batch_idx), self.T))
                    y_target = self.y[batch_idx + self.T]

                    # create X, y training batches
                    for k in range(len(batch_idx)):
                        # X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T-1), :]
                        # y_history[k, :] = self.y[batch_idx[k] : (batch_idx[k] + self.T-1)].flatten()
                        X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T), :]
                        y_history[k, :] = self.y[batch_idx[k] : (batch_idx[k] + self.T)].flatten()

                    # train
                    loss = self.train_iteration(X, y_target, y_history)
                    self.iter_losses[int(i * iter_per_epoch + j/self.batch_size)] = loss
                    if (j/self.batch_size) % 50 == 0:
                        print("\tbatch: %s loss: %s" %(j/self.batch_size, loss))
                        if self.logger != None:
                            self.logger.info("\tbatch: %s loss: %s" %(j/self.batch_size, loss))

                    j += self.batch_size
                    n_iter += 1

                    # decrease learning rate
                    if n_iter % 10000 == 0 and n_iter > 0:
                        for param_group in self.model_optimizer.param_groups:
                            param_group["lr"] = param_group["lr"] * 0.9

                self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch,
                                                                      (i+1) * iter_per_epoch)])
                if i % 10 == 0:
                    print("Epoch %s, loss: %s" %(i, self.epoch_losses[i]))
                    self.logger.info("Epoch %s, loss: %s" %(i, self.epoch_losses[i]))
                    # make prediction
                    print("\n Predict")
                    y_train_pred = self.predict(on_train=True)
                    y_test_pred = self.predict(on_train=False)

                    if plot_predict:
                        plt.figure()
                        plt.plot(range(1, 1+len(self.y)), self.y, label="True")
                        plt.plot(range(self.T, len(y_train_pred) + self.T), y_train_pred,
                                 label="Predicted - Train")
                        plt.plot(range(self.T + len(y_train_pred), len(self.y)+1),
                                 y_test_pred, label="Predicted - Test")
                        plt.legend(loc="upper left")
                        plt.show()

    def train_iteration(self, X, y_target, y_history, use_cuda=True):

        # zero gradient - original code placement
        self.model_optimizer.zero_grad()

        # define variables
        if use_cuda:
            Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())
            yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            yht = yht.unsqueeze(2)
            y_pred = self.model(Xt, yht)
            y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())
        else:
            Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu())
            yht = Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu())
            yht = yht.unsqueeze(2)
            y_pred = self.model(Xt, yht)
            y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cpu())

        # losstrain_test_model
        loss = self.loss_func(y_pred, y_true)
        loss.backward()

        # optimizer
        self.model_optimizer.step()

        return loss.item()

    def predict(self, on_train=False, use_cuda=True):

        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
            print("Predict TRAIN")
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)
            print("Predict TEST")

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T), :]
                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.T)].flatten()
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size)].flatten()

            if use_cuda:
                Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda())
                yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
                yht = yht.unsqueeze(2)
                # Xt (N, C, H, W)
                pred_cuda = self.model(Xt, yht).cuda()
                y_pred[i:(i + self.batch_size)] = pred_cuda.cpu().data.numpy()[:, 0]
            else:
                Xt = Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu())
                yht = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cpu())
                yht = yht.unsqueeze(2)
                # Xt (N, C, H, W)
                y_pred[i:(i + self.batch_size)] = self.model(Xt, yht).cpu().data.numpy()[:, 0]

            i += self.batch_size

        return y_pred
