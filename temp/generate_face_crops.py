# read frames and generate the face crops using either dlib or Google's mediapipe
# iterate through all the frames and generate bounding boxes
# the bounding boxes then need to be resized to a fixed dimension of 64x64
import os
import os.path as osp
import random
import subprocess
from glob import glob
from tqdm import tqdm
import argparse

import math

import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim
# import face_alignment
import dlib

dlib_dir = '/ssd_scratch/cvit/aditya1/baselines/dlib'
predictor_path = osp.join(dlib_dir, 'shape_predictor_5_face_landmarks.dat')
facerec_model_path = osp.join(dlib_dir, 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(facerec_model_path)

# return the bounding box 
def get_det(imgpath):
    img = dlib.load_rgb_image(imgpath)
    dets = detector(img, 1)
    # shape = sp(img, dets[0])
    # face_descriptor = facerec.compute_face_descriptor(img, shape)
    # # print(face_descriptor)
    # return face_descriptor
    try:
        if dets[0] is None:
            return None
    except:
        return None

    return dets[0]

def resize_frame(frame, resize_dim=64):
    h, w, _ = frame.shape

    if h > w:
        padw, padh = (h-w)//2, 0
    else:
        padw, padh = 0, (w-h)//2

    padded = cv2.copyMakeBorder(frame, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.resize(padded, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return padded

def write_frame(frame, image_path):
    cv2.imwrite(image_path, frame)

if __name__ == '__main__':
    base_dir = '/ssd_scratch/cvit/aditya1/metrics_baseline/target_gt'
    
    # for all the images, generate their corresponding crops and resize frame and save

    input_dirs = list(set(glob(base_dir + '/*')) - set(glob(base_dir + '/*.mp4')))

    for input_dir in tqdm(input_dirs):
        print(f'Processing : {input_dir}')
        images = glob(input_dir + '/*.png')

        cropped_images = list()
        min_left, max_right, min_top, max_bottom = 256, 0, 256, 0

        for index, image in enumerate(images):
            rectangle = get_det(image)

            if rectangle is None:
                continue

            min_left = min(min_left, rectangle.left())
            min_top = min(min_top, rectangle.top())
            max_right = max(max_right, rectangle.right())
            max_bottom = max(max_bottom, rectangle.bottom())

        # apply the constant crop across the images 
        for index, image in enumerate(images):
            img = cv2.imread(image)
            cropped = img[min_top:max_bottom, min_left:max_right]
            cropped_resized = resize_frame(cropped)

            cropped_path = osp.join(image.rsplit('/', 1)[0], osp.basename(image).split('.')[0] + '_cropped.png')
            write_frame(cropped_resized, cropped_path)