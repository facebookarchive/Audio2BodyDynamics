# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Code for transforming keypoints to fixed reference frame in order to isolate
motion due to music.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

MIN_VALID_PTS = 2  # The minimum number of valid points to be


def normalizePts(pts):
    N = pts.shape[0]
    cent = np.mean(pts, axis=0)
    ptsNorm = pts - cent
    sumOfPointDistancesFromOriginSquared = np.sum(np.power(ptsNorm[:, 0:2], 2))
    if sumOfPointDistancesFromOriginSquared > 0:
        scaleFactor = \
            np.sqrt(2 * N) / np.sqrt(sumOfPointDistancesFromOriginSquared)
    else:
        scaleFactor = 1

    ptsNorm = ptsNorm * scaleFactor

    normMtxInv = np.array([[1 / scaleFactor, 0, 0],
                           [0, 1 / scaleFactor, 0],
                           [cent[0], cent[1], 1]])

    return ptsNorm, normMtxInv


def transformPtsWithT(pts, T):
    if pts.ndim != 2:
        raise Exception("Must 2-D array")
    newPts = np.zeros(pts.shape)
    newPts[:, 0] = (T[0, 0] * pts[:, 0]) + (T[1, 0] * pts[:, 1]) + T[2, 0]
    newPts[:, 1] = (T[0, 1] * pts[:, 0]) + (T[1, 1] * pts[:, 1]) + T[2, 1]
    return newPts


def alignKeypoints(keypoints, reference=None, keyptstouse=None, confthresh=None):
    can_transform = (reference is not None) and (keyptstouse is not None)
    if can_transform:
        pts = keypoints[keyptstouse]
        valid_ind = np.where(pts[:, 2] > confthresh)[0]
        if (len(valid_ind) >= MIN_VALID_PTS):
            fixed_pts = reference[keyptstouse][valid_ind]
            valid_keyps = pts[valid_ind]
            try:
                transform = findNonreflectiveSimilarity(valid_keyps, fixed_pts)
                alignedKeypoints = transformPtsWithT(keypoints, transform)
            except Exception as e:
                print(e)
                transform = np.zeros((3, 3))
                transform[0, 0] = transform[1, 1] = 1
                alignedKeypoints = keypoints

    else:
        transform = np.zeros((3, 3))
        transform[0, 0] = transform[1, 1] = 1
        alignedKeypoints = keypoints
    return alignedKeypoints, transform


def findNonreflectiveSimilarity(src, dst):
    src, normMatrix1 = normalizePts(src)
    dst, normMatrix2 = normalizePts(dst)

    minRequiredNonCollinearPairs = 2
    M = dst.shape[0]

    x = np.expand_dims(dst[:, 0], axis=1)
    y = np.expand_dims(dst[:, 1], axis=1)
    X = np.concatenate((np.concatenate((x, y, np.ones((M, 1)), np.zeros((M, 1))), axis=1),
                        np.concatenate((y, -x, np.zeros((M, 1)), np.ones((M, 1))), axis=1)), axis=0)

    u = np.expand_dims(src[:, 0], axis=1)
    v = np.expand_dims(src[:, 1], axis=1)
    U = np.concatenate((u, v), axis=0)

    # We know that X * r = U
    if np.linalg.matrix_rank(X) >= 2 * minRequiredNonCollinearPairs:
        r, _, _, _ = np.linalg.lstsq(X, U)
    else:
        raise ValueError('images:geotrans:requiredNonCollinearPoints',
                         minRequiredNonCollinearPairs, 'nonreflectivesimilarity')

    sc = float(r[0])
    ss = float(r[1])
    tx = float(r[2])
    ty = float(r[3])

    Tinv = np.array([[sc, -ss, 0],
                     [ss, sc, 0],
                     [tx, ty, 1]])

    Tinv = np.linalg.solve(normMatrix2, np.dot(Tinv, normMatrix1))
    T = np.linalg.inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T


def buildRotT(sina):
    cosa = np.sqrt(1 - sina ** 2)
    T = np.array([[cosa, -sina, 0],
                  [sina, cosa, 0],
                  [0, 0, 1]])
    return T
