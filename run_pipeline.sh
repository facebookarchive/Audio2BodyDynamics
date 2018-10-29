#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# This script provides end-to-end training, testing and visualization on separate body parts

test_audio="data/test_audio.wav"

# Update Arguments with desired configuration

# Train on Body
echo "TRAINING MODEL ON BODY KEYPOINTS"
part="body"
# python pytorch_A2B_dynamics.py --data "data/train_$part.json" --vidtype body --numpca -1 --upsample_times 1 --logfldr "$HOME/logfldr/$part" --max_epoch 100 --time_steps 150
# Test on Body
echo "TESTING MODEL ON BODY KEYPOINTS"
python pytorch_A2B_dynamics.py --data "data/test_$part.json" --test_model "$HOME/logfldr/$part/best_model_db.pth" --vidtype "$part" --batch_size 1 --audio_file "$test_audio" --logfldr "$HOME/logfldr/$part"

# Train on Left Hand
echo "TRAINING MODEL ON LEFTHAND KEYPOINTS"
part="lefthand"
# python pytorch_A2B_dynamics.py --data "data/train_$part.json" --vidtype body --numpca 15 --upsample_times 1 --logfldr "$HOME/logfldr/$part" --max_epoch 100 --time_steps 150
# Test on Left Hand
echo "TESTING MODEL ON LEFTHAND KEYPOINTS"
python pytorch_A2B_dynamics.py --data "data/test_$part.json" --test_model "$HOME/logfldr/$part/best_model_db.pth" --vidtype "$part" --batch_size 1 --audio_file "$test_audio" --logfldr "$HOME/logfldr/$part"

# Train on Right Hand
echo "TRAINING MODEL ON RIGHTHAND KEYPOINTS"
part="righthand"
# python pytorch_A2B_dynamics.py --data "data/train_$part.json" --vidtype body --numpca 15 --upsample_times 1 --logfldr "$HOME/logfldr/$part" --max_epoch 100 --time_steps 150
# Test on Right Hand
echo "TESTING MODEL ON RIGHTHAND KEYPOINTS"
python pytorch_A2B_dynamics.py --data "data/test_$part.json" --test_model "$HOME/logfldr/$part/best_model_db.pth" --vidtype "$part" --batch_size 1 --audio_file "$test_audio" --logfldr "$HOME/logfldr/$part"

# Generate the Stitched Video
echo "GENERATING VIDEO OF STITCHED PART KEYPOINTS"
vidtype="piano"
python generate_stitched_video.py --vidtype "$vidtype" --body_path  "$HOME/logfldr/body/body_data.json" --righthand_path "$HOME/logfldr/righthand/righthand_data.json" --lefthand_path \
  "$HOME/logfldr/lefthand/lefthand_data.json" --gt_path "$HOME/logfldr/gt.mp4" --pred_path "$HOME/logfldr/pred.mp4" --audio_path "$test_audio"
