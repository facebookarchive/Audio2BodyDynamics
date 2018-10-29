# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
from sklearn.decomposition import PCA


class dataBatcher(object):

    def __init__(self, data, seq_len, batch_size, delay, shuffle=False):
        self.cur_batch = 0
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.delay = delay

        self.num_batches = len(self.data[0]) // (self.seq_len * self.batch_size)
        assert(self.num_batches != 0), 'Size of data must be > time_steps x batch_size'
        self.indices = range(self.num_batches)
        if shuffle:
            self.indices = np.random.permutation(self.num_batches)

    def hasNext(self):
        return self.cur_batch < len(self.indices)

    def correctDimensions(self, arr):
        num_feats = arr.shape[-1]
        arr = np.reshape(arr, [self.batch_size, self.seq_len, num_feats])
        arr = np.transpose(arr, (1, 0, 2))  # LSTM expects seqlen x batchsize x D
        return arr

    def delayArray(self, arr, dummy_var=0):
        arr[self.delay:, :, :] = arr[:(self.seq_len - self.delay), :, :]
        arr[:self.delay, :, :] = dummy_var
        return arr

    def reconstructKeypsOrder(self, arr):
        arr = np.reshape(arr, [self.seq_len, self.batch_size, -1])
        arr = arr[self.delay:, :, :]
        arr = np.transpose(arr, (1, 0, 2))
        num_pts = arr.shape[2]
        arr = np.reshape(arr, [-1, num_pts])
        arr = np.reshape(arr, [-1, 2, num_pts // 2])  # convert to X-Y format
        return arr

    def getNext(self):
        start = self.indices[self.cur_batch] * self.seq_len * self.batch_size
        end = (self.indices[self.cur_batch] + 1) * self.seq_len * self.batch_size
        assert((end - start) == (self.seq_len * self.batch_size))
        cur_aud = np.copy(self.data[0][start: end])
        cur_keyps = np.copy(self.data[1][start: end])

        cur_aud = self.correctDimensions(cur_aud)
        cur_keyps = self.correctDimensions(cur_keyps)
        cur_keyps = self.delayArray(cur_keyps)
        cur_keyps = np.reshape(cur_keyps, [-1, cur_keyps.shape[2]])

        mask = np.ones((self.seq_len * self.batch_size, 1))
        mask = self.correctDimensions(mask)
        mask = self.delayArray(mask)
        mask = np.reshape(mask, [-1, mask.shape[2]])

        self.cur_batch += 1
        return cur_aud, cur_keyps, mask


class DataIterator(object):

    def __init__(self, args, data_loc, test_mode=False):
        super(DataIterator, self).__init__()
        self.test_mode = test_mode
        self.seq_len = args.time_steps
        if self.test_mode:
            assert(args.batch_size == 1), \
                'No batching at test time. Run on full sequence.'
        self.batch_sz = args.batch_size
        self.delay = args.time_delay
        val_split = args.val_split if not self.test_mode else 1.0
        self.loadData(data_loc, val_split, args.upsample_times)
        if not self.test_mode:
            if (args.numpca > 0):
                self.performPCA(args.numpca)
            else:
                self.pca = None
            self.getDataStats()
            self.normalizeDataset()

    def stateDict(self):
        state_dict = {}
        state_dict['pca'] = self.pca
        state_dict['audio_stats'] = (self.aud_means, self.aud_stds)
        state_dict['keyps_stats'] = (self.means, self.stds)
        return state_dict

    def loadStateDict(self, state_dict):
        self.pca = state_dict['pca']
        self.aud_means, self.aud_stds = state_dict['audio_stats']
        self.means, self.stds = state_dict['keyps_stats']
        return state_dict

    def getPCASeq(self, seq, pca_dim=0, batch_dim=0):
        seq = np.reshape(seq, [self.seq_len, self.batch_sz, -1])
        seq = seq[self.delay:, :, :]
        seq = np.transpose(seq, (1, 0, 2))
        return seq[batch_dim, : , pca_dim]

    def processTestData(self, upsample_times=1):
        if self.pca:
            self.val_keyps = self.pca.transform(self.val_keyps)
        self.normalizeDataset()

    def loadData(self, data_loc, val_split, upsample_times):
        with open(data_loc, "r+") as fhandle:
            data = json.load(fhandle)
        self.train_audio, self.train_keyps = [], []
        self.val_audio, self.val_keyps = [], []

        # Data Format : video_id : (audio_features, body_keypoints, raw_audio)
        # body keypoints may or may not be transformed into fixed reference depending
        # depending on user.
        num_pts_per_batch = self.seq_len

        for _, (audio_feats, keyps) in data.items():
            audio_feats = np.array(audio_feats)
            keyps = np.array(keyps)

            if (upsample_times > 0):
                audio_feats = self.upsample(audio_feats, upsample_times)
                keyps = self.upsample(keyps, upsample_times)

            if not self.test_mode:
                # In Training mode, split the data
                num_batches = len(audio_feats) // num_pts_per_batch
                num_train = int(num_batches * (1 - val_split))

                # Throw away extra points
                audio_feats = audio_feats[:(num_batches * num_pts_per_batch)]
                keyps = keyps[:(num_batches * num_pts_per_batch)]

                audio_split = np.split(audio_feats, num_batches)
                keyps_split = np.split(keyps, num_batches)

                # Split into Train and Test
                train_indices = np.random.choice(num_batches, num_train, replace=False)
                val_indices = [x for x in range(num_batches) if x not in train_indices]

                for ind in train_indices:
                    assert(len(audio_split[ind]) == num_pts_per_batch)
                    self.train_audio.extend(audio_split[ind])
                    self.train_keyps.extend(keyps_split[ind])

                for ind in val_indices:
                    assert(len(keyps_split[ind]) == num_pts_per_batch)
                    self.val_audio.extend(audio_split[ind])
                    self.val_keyps.extend(keyps_split[ind])
            else:
                self.val_audio = audio_feats
                self.val_keyps = keyps

                # Perform Inference on whole video at once
                self.seq_len = len(self.val_keyps)

                break  # Testing is executed on 1 video at a time.

        self.train_audio, self.train_keyps = \
            np.array(self.train_audio), np.array(self.train_keyps)
        self.val_audio, self.val_keyps = \
            np.array(self.val_audio), np.array(self.val_keyps)

    def performPCA(self, num_components):
        self.pca = PCA(n_components=num_components)
        self.train_keyps = self.pca.fit_transform(self.train_keyps)
        self.val_keyps = self.pca.transform(self.val_keyps)

    def getDataStats(self):
        self.means = self.train_keyps.mean(axis=0)
        self.stds = np.max(self.train_keyps.std(axis=0))

        self.aud_means = 0.0
        self.aud_stds = 1.0

    def normalizeDataset(self):

        def normalize(dataset, mean, std):
            EPSILON = 1E-8
            if not len(dataset):
                return dataset
            return (dataset - mean) / (std + EPSILON)

        self.train_keyps = normalize(self.train_keyps, self.means, self.stds)
        self.val_keyps = normalize(self.val_keyps, self.means, self.stds)

        self.train_audio = normalize(self.train_audio, self.aud_means, self.aud_stds)
        self.val_audio = normalize(self.val_audio, self.aud_means, self.aud_stds)

    def getInOutDimensions(self):
        return self.val_audio.shape[1], self.val_keyps.shape[1]

    def reset(self):
        self.val_iterator = self.createIterator(is_test=True)
        if not self.test_mode:
            self.train_iterator = self.createIterator(is_test=False)

    def getNumBatches(self):
        train_batches = len(self.train_keyps) // (self.seq_len * self.batch_sz)
        val_batches = len(self.val_keyps) // (self.seq_len * self.batch_sz)
        return train_batches, val_batches

    def createIterator(self, is_test=False):
        dataset = (self.val_audio, self.val_keyps) \
            if is_test else (self.train_audio, self.train_keyps)
        return dataBatcher(dataset, self.seq_len, self.batch_sz, self.delay,
                           shuffle=(not is_test))

    def hasNext(self, is_test=False):
        if is_test:
            return self.val_iterator.hasNext()
        else:
            return self.train_iterator.hasNext()

    def nextBatch(self, is_test=False):
        if is_test:
            return self.val_iterator.getNext()
        else:
            return self.train_iterator.getNext()

    def reconstructKeypsOrder(self, batch):
        return self.val_iterator.reconstructKeypsOrder(batch)

    def reconstructAudioOrder(self, batch):
        return self.val_iterator.reconstructAudioOrder(batch)

    def toPixelSpace(self, predictions):
        recon = (predictions * self.stds) + self.means
        if self.pca:
            recon = self.pca.inverse_transform(recon)
        return recon

    def upsample(self, array, n_times, use_repeat=False):
        if not len(array):
            return array
        result = array
        for _ in range(n_times):
            if use_repeat:
                result = np.repeat(result, 2, axis=0)
                result = result[:-1]
            else:
                n_examples, n_feats = result.shape
                new_arry = np.zeros((n_examples * 2 - 1, n_feats))
                new_arry[0::2, :] = result
                new_arry[1::2, :] = (result[1:, :] + result[:-1, :]) / 2.0
                result = new_arry
        return result
