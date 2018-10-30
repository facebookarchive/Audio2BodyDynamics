# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
from visualize import visualizeKeypoints
import argparse

'''
Example Run

python generateStitchedVideo.py --vidtype piano --body_path final_body/body_data.json
    --righthand_path final_righthand/righthand_data.json
    --lefthand_path final_lefthand/lefthand_data.json
    --vid_path viz/all_pts.mp4 --pred_path viz/just_pred.mp4
    --audio_path audio.wav
'''


def stitchHandsToBody(body_keyps, rh_keyps, lh_keyps):
    body_keyps = np.array(body_keyps)
    rh_keyps = np.array(rh_keyps)
    lh_keyps = np.array(lh_keyps)
    for ind, pts in enumerate(body_keyps):
        rh_diff = np.expand_dims(pts[:, 2] - rh_keyps[ind][:, 0], axis=1)
        lh_diff = np.expand_dims(pts[:, 5] - lh_keyps[ind][:, 0], axis=1)
        rh_keyps[ind] = rh_keyps[ind] + rh_diff
        lh_keyps[ind] = lh_keyps[ind] + lh_diff
    return body_keyps, rh_keyps, lh_keyps


def createOptions():
    # Default configuration for PianoNet
    parser = argparse.ArgumentParser(
        description="Pytorch: Audio To Body Dynamics Model"
    )
    parser.add_argument("--body_path", type=str, default="body_data.json",
                        help="Path to body keypoints")
    parser.add_argument("--righthand_path", type=str, default="righthand_data.json",
                        help="Path to righthand keypoints")
    parser.add_argument("--lefthand_path", type=str, default="lefthand_data.json",
                        help="Path to righthand keypoints")
    parser.add_argument("--gt_path", type=str, default="ground_truth.mp4",
                        help="Where to save the ground_truth video.")
    parser.add_argument("--pred_path", type=str, default="predictions.mp4",
                        help="Where to save the resulting prediction video.")
    parser.add_argument("--vidtype", type=str, default='piano',
                        help="Type of video whether piano or violin")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Only in for Test. Location audio file for"
                             " generating test video")
    args = parser.parse_args()
    return args


def main():
    args = createOptions()
    body = json.load(open(args.body_path, 'r+'))
    righthand = json.load(open(args.righthand_path, 'r+'))
    lefthand = json.load(open(args.lefthand_path, 'r+'))

    all_pred_pts = np.concatenate(stitchHandsToBody(body[1], righthand[1],
                                  lefthand[1]), axis=2)
    all_targ_pts = np.concatenate((body[0], righthand[0], lefthand[0]), axis=2)

    # Just Gt
    visualizeKeypoints(args.vidtype, all_targ_pts, all_pred_pts,
                       args.audio_path, args.gt_path, show_pred=False)

    # Just Pred
    visualizeKeypoints(args.vidtype, all_targ_pts, all_pred_pts, args.audio_path,
                       args.pred_path, show_gt=False)


if __name__ == '__main__':
    main()
