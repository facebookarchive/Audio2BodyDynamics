# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import os
import numpy as np
from data_utils.transform_keypoints import transformPtsWithT
from scipy.io.wavfile import write

THRESH = 0
FFMPEG_LOC = "ffmpeg "


def getUpperOPBodyKeypsLines():
    kp_lines = [[0, 1], [1, 2], [0, 3], [3, 4], [4, 5]]
    return kp_lines


def getUpperOPSHELBodyKeypsLines():
    kp_lines = [[0, 1], [0, 2], [2, 3]]
    return kp_lines


def getUpperOPELWRBodyKeypsLines():
    kp_lines = [[0, 1], [2, 3]]
    return kp_lines


def getUpperOPWRBodyKeypsLines():
    kp_lines = [[0, 1]]
    return kp_lines


def getUpperOPONESIDEBodyKeypsLines():
    kp_lines = [[0, 1], [1, 2]]
    return kp_lines


def getUpperOPHandsKeypsLines():
    kp_lines = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5],
                [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
                [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    return kp_lines


def drawMouth(image, mouthlmk, color=(0, 255, 0)):
    x = mouthlmk[:, 0]
    y = mouthlmk[:, 1]
    for indices in range(len(x) - 1):
        x1, x2 = int(x[indices]), int(x[indices + 1])
        y1, y2 = int(y[indices]), int(y[indices + 1])
        if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
            pt1, pt2 = (x1, y1), (x2, y2)
            cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)
    return image


def drawBody(image, bodylmk, tform=None, confidences=None, color=(0, 255, 0),
             diffx=-750, diffy=-100):
    lines = getUpperOPBodyKeypsLines()
    x = bodylmk[0]
    y = bodylmk[1]
    for indices in lines:
        # incorporating tform
        if confidences[indices[0]] < THRESH or confidences[indices[1]] < THRESH:
            continue

        x1, x2 = int(x[indices[0]]), int(x[indices[1]])
        y1, y2 = int(y[indices[0]]), int(y[indices[1]])

        if tform is not None:

            tpts = transformPtsWithT(np.array([[x1, y1], [x2, y2]]), tform)
            x1, x2 = int(tpts[0, 0]), int(tpts[1, 0])
            y1, y2 = int(tpts[0, 1]), int(tpts[1, 1])

        # pdb.set_trace()
        x1, x2 = x1 + diffx, x2 + diffx
        y1, y2 = y1 + diffy, y2 + diffy

        if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
            pt1, pt2 = (x1, y1), (x2, y2)
            cv2.circle(image, pt1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(image, pt2, 4, color, -1, cv2.LINE_AA)
            cv2.line(image, pt1, pt2, color, 3, cv2.LINE_AA)
    return image


def drawBodyAndFingers(vidtype, image, bodylmk, tform=None, confidences=None,
                       color=(0, 255, 0)):

    if vidtype == 'shouldelbows':
        lines = getUpperOPSHELBodyKeypsLines()
    elif vidtype == 'elbowswrists':
        lines = getUpperOPELWRBodyKeypsLines()
    elif vidtype == 'wrists' or vidtype == 'vshould':
        lines = getUpperOPWRBodyKeypsLines()
    elif vidtype == 'righthand' or vidtype == 'vrighthand':
        lines = []
    elif vidtype == 'lefthand' or vidtype == 'vlefthand':
        lines = []
    elif vidtype == 'violinleft' or vidtype == 'violinright':
        lines = getUpperOPONESIDEBodyKeypsLines()
    else:
        lines = getUpperOPBodyKeypsLines()

    x = bodylmk[0]
    y = bodylmk[1]
    for indices in lines:

        x1, x2 = int(x[indices[0]]), int(x[indices[1]])
        y1, y2 = int(y[indices[0]]), int(y[indices[1]])

        if tform is not None:
            tpts = transformPtsWithT(np.array([[x1, y1], [x2, y2]]), tform)
            x1, x2 = int(tpts[0, 0]), int(tpts[1, 0])
            y1, y2 = int(tpts[0, 1]), int(tpts[1, 1])

        if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
            pt1, pt2 = (x1, y1), (x2, y2)
            cv2.circle(image, pt1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(image, pt2, 3, color, -1, cv2.LINE_AA)
            cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

    if vidtype == 'violin':
        shft = 8
        hlines = np.array(getUpperOPHandsKeypsLines()) + shft
        hlines = np.append(hlines, np.array(getUpperOPHandsKeypsLines()) + 21 + shft, 0)
    elif vidtype == 'piano':
        shft = 7
        hlines = np.array(getUpperOPHandsKeypsLines()) + shft
        hlines = np.append(hlines, np.array(getUpperOPHandsKeypsLines()) + 21 + shft, 0)
    elif vidtype == 'righthand' or vidtype == 'lefthand' or \
         vidtype == 'vrighthand' or vidtype == 'vlefthand':
        shft = 0
        hlines = np.array(getUpperOPHandsKeypsLines()) + shft
    else:
        return image

    for indices in hlines:
        x1, x2 = int(x[indices[0]]), int(x[indices[1]])
        y1, y2 = int(y[indices[0]]), int(y[indices[1]])

        if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
            pt1, pt2 = (x1, y1), (x2, y2)
            cv2.circle(image, pt1, 2, color, -1, cv2.LINE_AA)
            cv2.circle(image, pt2, 2, color, -1, cv2.LINE_AA)
            cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)
    return image


def writeAudio(vid_loc, audio_loc):
    new_vid_loc = vid_loc.split(".mp4")[0] + "_audio.mp4"
    cmd = FFMPEG_LOC + " -loglevel panic -i " + vid_loc + " -i " + audio_loc
    cmd += " -c:v copy -c:a aac -strict experimental " + new_vid_loc
    os.system(cmd)
    return new_vid_loc


def videoFromImages(imgs, outputfile, audio_path, fps=27.1):
    fourcc_format = cv2.VideoWriter_fourcc(*'MP4V')
    size = imgs[0].shape[1], imgs[0].shape[0]
    vid = cv2.VideoWriter(outputfile, fourcc_format, fps, size)
    for img in imgs:
        vid.write(img)
    vid.release()
    if audio_path is not None:
        writeAudio(outputfile, audio_path)
    return outputfile


def visualizeKeypoints(vidtype, targetKeypts, predictedKeypts, audio_path,
                        outfile, img_size=300, show_pred=True, show_gt=True, fps=27.1):
        targ_color = (0, 255, 0)
        pred_color = (0, 0, 255)
        images = []

        for ind, targkeyps in enumerate(targetKeypts):
            img_y, img_x = (img_size, img_size)
            if show_gt and show_pred:
                img_y, img_x = (img_size, img_size * 2)
                targkeyps[0] += img_size  # Shift the ground truth points

            newImage = np.ones((img_y, img_x, 3), dtype=np.uint8) * 255
            predkeyps = predictedKeypts[ind]
            if show_pred:
                newImage = drawBodyAndFingers(vidtype, newImage, predkeyps,
                                              color=pred_color)
            if show_gt:
                newImage = drawBodyAndFingers(vidtype, newImage, targkeyps,
                                              color=targ_color)

            newImage = cv2.resize(newImage, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_CUBIC)

            images.append(newImage)
        if images:
            videoFromImages(images, outfile, audio_path, fps=fps)
