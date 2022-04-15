# method used to verify the dlib score between two pair of images
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
import dlib

BASE_DIR = '/ssd_scratch/cvit/aditya1/metrics_baseline/faceoff_results'

pred_images = glob(BASE_DIR + '/*/*.png')


pred_image = pred_images[random.randint(0, len(pred_images)-1)]
gt_image = pred_image.replace('faceoff_results', 'source_gt').replace('_prediction', '_source')

dlib_dir = '/ssd_scratch/cvit/aditya1/baselines/dlib'
predictor_path = osp.join(dlib_dir, 'shape_predictor_5_face_landmarks.dat')
facerec_model_path = osp.join(dlib_dir, 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(facerec_model_path)

def get_det(imgpath):
    img = dlib.load_rgb_image(imgpath)
    dets = detector(img, 1)
    shape = sp(img, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    # print(face_descriptor)
    return face_descriptor

det1 = np.array(get_det(pred_image))
print(det1)
det2 = np.array(get_det(gt_image))

# compute the euclidean distance between the two images 
distance = np.linalg.norm(det2 - det1)
print(f'Values : {distance, pred_image, gt_image}')