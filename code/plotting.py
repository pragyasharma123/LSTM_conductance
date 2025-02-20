# Author: Laura Kulowski

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, num_rows=4):
    '''
    Plot examples of the LSTM encoder-decoder evaluated on the training/test data.

    :param lstm_model:     Trained LSTM encoder-decoder
    :param Xtrain:         np.array of windowed training input data
    :param Ytrain:         np.array of windowed training target data
    :param Xtest:          np.array of windowed test input data
    :param Ytest:          np.array of windowed test target data 
    :param num_rows:       Number of training/test examples to plot
    :return:               num_rows x 2 plots; first column is training data predictions,
                           second column is test data predictions
    '''

    # Input window size
    iw = Xtrain.shape[0]
    ow = Ytest.shape[0]

    # Figure setup 
    num_cols = 2
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(13, 15))

    # Get the model's device
    device = next(lstm_model.parameters()).device

    # Plot training/test predictions
    for ii in range(num_rows):
        # Train set
        X_train_plt = torch.from_numpy(Xtrain[:, ii, :]).type(torch.Tensor).to(device)
        Y_train_pred = lstm_model.predict(X_train_plt, target_len=ow)
        Y_train_pred = Y_train_pred  # Already a NumPy array, no need to detach

        ax[ii, 0].plot(np.arange(0, iw), Xtrain[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow),  np.concatenate([[Xtrain[-1, ii, 0]], Y_train_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 0].set_xlim([0, iw + ow - 1])
        ax[ii, 0].set_xlabel('$t$')
        ax[ii, 0].set_ylabel('$y$')

        # Test set
        X_test_plt = torch.from_numpy(Xtest[:, ii, :]).type(torch.Tensor).to(device)
        Y_test_pred = lstm_model.predict(X_test_plt, target_len=ow)

        ax[ii, 1].plot(np.arange(0, iw), Xtest[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 1].set_xlim([0, iw + ow - 1])
        ax[ii, 1].set_xlabel('$t$')
        ax[ii, 1].set_ylabel('$y$')

        if ii == 0:
            ax[ii, 0].set_title('Train')
            ax[ii, 1].legend(bbox_to_anchor=(1, 1))
            ax[ii, 1].set_title('Test')

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('plots/predictions.png')
    plt.close()

    return