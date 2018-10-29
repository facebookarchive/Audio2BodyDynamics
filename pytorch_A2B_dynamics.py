# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import optim
from model import AudioToKeypointRNN
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import argparse
import logging
import numpy as np
import json
from data_utils.data import DataIterator
from visualize import visualizeKeypoints

'''
This script takes an input of audio MFCC features and uses
an LSTM recurrent neural network to learn to predict
body joints coordinates
'''

logging.basicConfig()
log = logging.getLogger("mannerisms_rnn")
log.setLevel(logging.DEBUG)
torch.manual_seed(1234)
np.random.seed(1234)


class AudoToBodyDynamics(object):

    def __init__(self, args, data_locs, is_test=False):
        super(AudoToBodyDynamics, self).__init__()

        self.is_test_mode = is_test
        self.data_iterator = DataIterator(args, data_locs, test_mode=is_test)

        # Refresh data configuration from checkpoint
        if self.is_test_mode:
            self.loadDataCheckpoint(args.test_model, args.upsample_times)

        input_dim, output_dim = self.data_iterator.getInOutDimensions()

        # construct the model
        model_options = {
            'device': args.device,
            'dropout': args.dp,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_size,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'trainable_init': args.trainable_init
        }
        self.vidtype = args.vidtype
        self.device = args.device
        self.log_frequency = args.log_frequency
        self.upsample_times = args.upsample_times
        self.model = AudioToKeypointRNN(model_options).cuda(args.device)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)

        # Load checkpoint model
        if self.is_test_mode:
            self.loadModelCheckpoint(args.test_model)

    def buildLoss(self, rnn_out, target, mask):
        square_diff = (rnn_out - target)**2
        out = torch.sum(square_diff, 1, keepdim=True)
        masked_out = out * mask
        return torch.mean(masked_out), masked_out

    def saveModel(self, state_info, path):
        torch.save(state_info, path)

    def loadModelCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def loadDataCheckpoint(self, path, upsample_times):
        checkpoint = torch.load(path)
        self.data_iterator.loadStateDict(checkpoint['data_state_dict'])
        self.data_iterator.processTestData(upsample_times=upsample_times)

    def runNetwork(self, validate=False):
        def to_numpy(x):
            return x.cpu().data.numpy()

        # Set up inputs to the network
        batch_info = self.data_iterator.nextBatch(is_test=validate)
        in_batch, out_batch, mask_batch = batch_info
        inputs = Variable(torch.FloatTensor(in_batch).to(self.device))
        targets = Variable(torch.FloatTensor(out_batch).to(self.device))
        masks = Variable(torch.FloatTensor(mask_batch).to(self.device))

        # Run the network
        predictions = self.model.forward(inputs)

        # Get loss in pca coefficient space
        loss, _ = self.buildLoss(predictions, targets, masks)

        # Get loss in pixel space
        pixel_predictions = self.data_iterator.toPixelSpace(to_numpy(predictions))
        pixel_predictions = torch.FloatTensor(pixel_predictions).to(self.device)

        pixel_targets = self.data_iterator.toPixelSpace(out_batch)
        pixel_targets = torch.FloatTensor(pixel_targets).to(self.device)
        _, frame_loss = self.buildLoss(pixel_predictions, pixel_targets, masks)

        frame_loss = frame_loss / pixel_targets.size()[1]
        # Gives the average deviation of prediction from target pixel
        pixel_loss = torch.mean(torch.sqrt(frame_loss))

        return (to_numpy(predictions), to_numpy(targets)), loss, pixel_loss

    def runEpoch(self):
        pixel_losses, coeff_losses = [], []
        val_pix_losses, val_coeff_losses = [], []
        predictions, targets = [], []

        while (not self.is_test_mode and self.data_iterator.hasNext(is_test=False)):
            self.model.train()
            _, pca_coeff_loss, pixel_loss = self.runNetwork(validate=False)
            self.optim.zero_grad()
            pca_coeff_loss.backward()
            self.optim.step()

            pca_coeff_loss = pca_coeff_loss.data.tolist()
            pixel_loss = pixel_loss.data.tolist()
            pixel_losses.append(pixel_loss)
            coeff_losses.append(pca_coeff_loss)

        while(self.data_iterator.hasNext(is_test=True)):
            self.model.eval()
            vis_data, pca_coeff_loss, pixel_loss = self.runNetwork(validate=True)
            pca_coeff_loss = pca_coeff_loss.data.tolist()
            pixel_loss = pixel_loss.data.tolist()

            val_pix_losses.append(pixel_loss)
            val_coeff_losses.append(pca_coeff_loss)

            predictions.append(vis_data[0])
            targets.append(vis_data[1])

        train_info = (pixel_losses, coeff_losses)
        val_info = (val_pix_losses, val_coeff_losses)
        return train_info, val_info, predictions, targets

    def trainModel(self, max_epochs, logfldr, patience):
        log.debug("Training model")
        epoch_losses = []
        batch_losses = []
        val_losses = []
        i, best_loss, iters_without_improvement = 0, float('inf'), 0
        best_train_loss, best_val_loss = float('inf'), float('inf')

        while(i < max_epochs):
            i += 1
            self.data_iterator.reset()
            iter_train, iter_val, predictions, targets = self.runEpoch()
            iter_mean = np.mean(iter_train[0]), np.mean(iter_train[1])
            iter_val_mean = np.mean(iter_val[0]), np.mean(iter_val[1])

            epoch_losses.append(iter_mean)
            batch_losses.extend(iter_train)
            val_losses.append(iter_val_mean)

            log.info("Epoch {} / {}".format(i, max_epochs))
            log.info("Training Loss (1980 x 1080): {}".format(iter_mean))
            log.info("Validation Loss (1980 x 1080): {}".format(iter_val_mean))

            improved = iter_val_mean[1] < best_loss
            if improved:
                best_loss = iter_val_mean[1]
                best_val_loss = iter_val_mean
                best_train_loss = iter_mean
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1
                if iters_without_improvement >= patience:
                    log.info("Stopping Early because no improvment in {}".format(
                        iters_without_improvement))
                    break
            if improved or (i % self.log_frequency) == 0:
                # Save the model information
                path = os.path.join(logfldr, "Epoch_{}".format(i))
                os.makedirs(path)
                path = os.path.join(path, "model_db.pth")
                state_info = {
                    'epoch': i,
                    'epoch_losses': epoch_losses,
                    'batch_losses': batch_losses,
                    'validation_losses': val_losses,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'data_state_dict': self.data_iterator.stateDict()
                }
                self.saveModel(state_info, path)
                if improved:
                    path = os.path.join(logfldr, "best_model_db.pth")
                    self.saveModel(state_info, path)

                # Visualize the PCA Coefficients
                num_vis = min(3, targets[0].shape[-1])
                for j in range(num_vis):
                    save_path = os.path.join(
                        logfldr, "Epoch_{}/pca_{}.png".format(i, j))
                    self.visualizePCA(predictions[0], targets[0], j, save_path)

        self.plotResults(logfldr, epoch_losses, batch_losses, val_losses)
        return best_train_loss, best_val_loss

    def formatVizArrays(self, predictions, targets):
        final_pred, final_targ = [], []
        for ind, pred in enumerate(predictions):
            pred = self.data_iterator.toPixelSpace(pred)
            targ = self.data_iterator.toPixelSpace(targets[ind])
            pred = self.data_iterator.reconstructKeypsOrder(pred)
            targ = self.data_iterator.reconstructKeypsOrder(targ)
            final_pred.append(pred)
            final_targ.append(targ)

        final_pred, final_targ = np.vstack(final_pred), np.vstack(final_targ)
        final_pred = final_pred[0::(2**self.upsample_times)]
        final_targ = final_targ[0::(2**self.upsample_times)]

        return final_pred, final_targ

    def visualizePCA(self, preds, targets, pca_dim, save_path):
        preds = self.data_iterator.getPCASeq(preds, pca_dim=pca_dim)
        targs = self.data_iterator.getPCASeq(targets, pca_dim=pca_dim)
        assert(len(preds) == len(targs))
        plt.plot(preds, color='red', label='Predictions')
        plt.plot(targs, color='green', label='Ground Truth')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plotResults(self, logfldr, epoch_losses, batch_losses, val_losses):
        losses = [epoch_losses, batch_losses, val_losses]
        names = [
            ["Epoch pixel losses", "Epoch coeff losses"],
            ["Batch pixel losses", "Batch coeff losses"],
            ["Val pixel losses", "Val coeff losses"]]
        _, ax = plt.subplots(nrows=len(losses), ncols=2)
        for index, pair in enumerate(zip(losses, names)):
            for i in range(2):
                data = [pair[0][j][i] for j in range(len(pair[0]))]
                ax[index][i].plot(data, label=pair[1][i])
                ax[index][i].legend()
        save_filename = os.path.join(logfldr, "results.png")
        plt.savefig(save_filename)
        plt.close()


def createOptions():
    # Default configuration for PianoNet
    parser = argparse.ArgumentParser(
        description="Pytorch: Audio To Body Dynamics Model"
    )
    parser.add_argument("--data", type=str, default="piano_data.json",
                        help="Path to data file")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Only in for Test. Location audio file for"
                             " generating test video")
    parser.add_argument("--logfldr", type=str, default=None,
                        help="Path to folder to save training information",
                        required=True)
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Training batch size. Set to 1 in test")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="The fraction of the training data to use as validation")
    parser.add_argument("--hidden_size", type=int, default=200,
                        help="Dimension of the hidden representation")
    parser.add_argument("--test_model", type=str, default=None,
                        help="Location for saved model to load")
    parser.add_argument("--vidtype", type=str, default='piano',
                        help="Type of video whether piano or violin")
    parser.add_argument("--visualize", type=bool, default=False,
                        help="Visualize the output of the model. Use only in Test")
    parser.add_argument("--save_predictions", type=bool, default=True,
                        help="Whether or not to save predictions. Use only in Test")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to train on. Use 'cpu' if to train on cpu.")
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="max number of epochs to run for")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning Rate for optimizer")
    parser.add_argument("--time_steps", type=int, default=60,
                        help="Prediction Timesteps")
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no validation improvement"
                        " before stopping training")
    parser.add_argument("--time_delay", type=int, default=6,
                        help="Time delay for RNN. Negative values mean no delay."
                        "Give in terms of frames. 30 frames = 1 second.")
    parser.add_argument("--dp", type=float, default=0.1,
                        help="Dropout Ratio For Trainining")
    parser.add_argument("--upsample_times", type=int, default=2,
                        help="number of times to upsample")
    parser.add_argument("--numpca", type=int, default=15,
                        help="number of pca dimensions. Use -1 if no pca - "
                             "Train on XY coordinates")
    parser.add_argument("--log_frequency", type=int, default=10,
                        help="The frequency with which to checkpoint the model")
    parser.add_argument("--trainable_init", action='store_false',
                        help="LSTM initial state should be trained. Default is True")

    args = parser.parse_args()
    return args


def main():
    args = createOptions()
    args.device = torch.device(args.device)
    data_loc = args.data
    is_test_mode = args.test_model is not None

    dynamics_learner = AudoToBodyDynamics(args, data_loc, is_test=is_test_mode)
    logfldr = args.logfldr
    if not os.path.isdir(logfldr):
        os.makedirs(logfldr)

    if not is_test_mode:
        min_train, min_val = dynamics_learner.trainModel(
            args.max_epochs, logfldr, args.patience)
    else:
        dynamics_learner.data_iterator.reset()
        outputs = dynamics_learner.runEpoch()
        iter_train, iter_val, targ, pred = outputs
        min_train, min_val = np.mean(iter_train[0]), np.mean(iter_val[0])

        # Format the visualization appropriately
        targ, pred = dynamics_learner.formatVizArrays(pred, targ)

        # Save the predictions
        if args.save_predictions:
            viz_info = (targ.tolist(), pred.tolist())
            save_path = "{}/{}_data.json".format(logfldr, args.vidtype)
            json.dump(viz_info, open(save_path, 'w+'))

        # Create Video of Results
        if args.visualize:
            vid_path = "{}/{}.mp4".format(logfldr, args.vidtype)
            visualizeKeypoints(args.vidtype, targ, pred, args.audio_file, vid_path)

    best_lossess = [min_train, min_val]
    log.info("The best validation is : {}".format(best_lossess))


if __name__ == '__main__':
    main()
