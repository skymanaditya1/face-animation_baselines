# Script used for computing different quantitative metrics 
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

# psnr metric 

# ssim metric 

# landmark distance metric 

# temporal metric 

# fid metric 

# gpu_id = 0
# fa = face_alignment.FaceAlignment(
#     face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(gpu_id))

dlib_dir = '/ssd_scratch/cvit/aditya1/baselines/dlib'
predictor_path = osp.join(dlib_dir, 'shape_predictor_5_face_landmarks.dat')
facerec_model_path = osp.join(dlib_dir, 'dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(facerec_model_path)

def read_frames(video_path):
	video_stream = cv2.VideoCapture(video_path)
	ret, frame = video_stream.read()
	frames = list()

	while ret:
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		ret, frame = video_stream.read()

	return frames

def save_frame(frame, filepath, require_color_conversion=True):
    if require_color_conversion:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, frame)

def save_frames(frames, save_dir):
    for index, frame in enumerate(frames):
        filepath = osp.join(save_dir, str(index).zfill(3) + '.png')
        save_frame(frame, filepath)


def _rgb2ycbcr(img, maxVal=255):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
	O = np.array([[16],
				  [128],
				  [128]])
	T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
				  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
				  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

	if maxVal == 1:
		O = O / 255.0

	t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
	t = np.dot(t, np.transpose(T))
	t[:, 0] += O[0]
	t[:, 1] += O[1]
	t[:, 2] += O[2]
	ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

	return ycbcr

def to_uint8(x, vmin, vmax):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
	x = x.astype('float32')
	x = (x-vmin)/(vmax-vmin)*255 # 0~255
	return np.clip(np.round(x), 0, 255)

# method to return the dlib feature embedding (128 dimensional)
def dlib_feature_embedding(imagepath):
    img = dlib.load_rgb_image(imagepath)
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = sp(img, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

# measures the euclidean distance between two face descriptors
def dlib_euclidean_distance(fd1, fd2):
    return np.linalg.norm(fd1 - fd2)

def psnr(img_true, img_pred):
##### PSNR with color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
	Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
	Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
	diff =  Y_true - Y_pred
	rmse = np.sqrt(np.mean(np.power(diff,2)))
	eps = 1e-8
	return 20*np.log10(255./(rmse+eps))

def compute_psnr_video(pred_dir, gt_dir):
    current_pred_frames = sorted(glob(pred_dir + '/*.png'))
    current_gt_frames = sorted(glob(gt_dir + '/*.png'))

    total_psnr_video = list()
    for index, (pred_frame, gt_frame) in enumerate(zip(current_pred_frames, current_gt_frames)):
        pred = cv2.imread(pred_frame)[:, :, ::-1]
        gt = cv2.imread(gt_frame)[:, :, ::-1]

        # compute the psnr metric between the two frames
        psnr_metric = psnr(gt, pred)

        if math.isnan(psnr_metric) or math.isinf(psnr_metric):
            continue
        
        total_psnr_video.append(psnr_metric)

    return sum(total_psnr_video) / len(total_psnr_video)

def compute_ssim(img_true, img_pred): ##### SSIM ##### 
	Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
	Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
	# return measure.compare_ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())
	return ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())

def compute_ssim_video(pred_dir, gt_dir):
    current_pred_frames = sorted(glob(pred_dir + '/*.png'))
    current_gt_frames = sorted(glob(gt_dir + '/*.png'))

    total_ssim_video = list()
    for index, (pred_frame, gt_frame) in enumerate(zip(current_pred_frames, current_gt_frames)):
        pred = cv2.imread(pred_frame)[:, :, ::-1]
        gt = cv2.imread(gt_frame)[:, :, ::-1]

        # compute the psnr metric between the two frames
        ssim_metric = compute_ssim(gt, pred)

        if math.isnan(ssim_metric) or math.isinf(ssim_metric):
            continue
        
        total_ssim_video.append(ssim_metric)

    return sum(total_ssim_video) / len(total_ssim_video)

# computes the rotation of the face using the angle of the line connecting the eye centroids 
def compute_rotation(shape):
    # landmark coordinates corresponding to the eyes 
    lStart, lEnd = 36, 41
    rStart, rEnd = 42, 47

    # landmarks for the left and right eyes 
    leftEyePoints = shape[lStart:lEnd]
    rightEyePoints = shape[rStart:rEnd]

    # compute the center of mass for each of the eyes 
    leftEyeCenter = leftEyePoints.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePoints.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) 
    
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    
    dist = np.sqrt((dX ** 2) + (dY ** 2)) # this indicates the distance between the two eyes 
    
    return angle, eyesCenter, dist

# generate the new landmarks for the image 
def generate_landmarks(img):
    landmarks = fa.get_landmarks(img)

    if landmarks is not None and len(landmarks) > 0:
        return landmarks[0]
    
    return None

def landmark_alignment(source_image, source_landmarks, target_image, target_landmarks):
    # align the target image based on the source image 
    source_rotation, source_center, source_distance = compute_rotation(source_landmarks)
    target_rotation, target_center, target_distance = compute_rotation(target_landmarks)

    # apply the rotation on the target conditioned on the source 
    source_conditioned_target_rotation = target_rotation - source_rotation
    scaling = source_distance / target_distance

    height, width = 256, 256

    rotate_matrix = cv2.getRotationMatrix2D(center=target_center, 
                    angle=source_conditioned_target_rotation, scale=scaling)

    # calculate the translation component of the matrix M 
    rotate_matrix[0, 2] += (source_center[0] - target_center[0])
    rotate_matrix[1, 2] += (source_center[1] - target_center[1])

    # apply the transformation to the target image 
    target_image_transformed = cv2.warpAffine(target_image, 
                                rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)

    # compute the transformed landmarks of the target 
    target_landmarks_transformed = generate_landmarks(target_image_transformed)
    target_rotation_t, target_center_t, target_distance_t = compute_rotation(target_landmarks_transformed)

    # print(source_center, target_center)
    # print(source_center, target_center_t)

    return target_landmarks_transformed

# computes the averaged EMD between landmark1 and landmark2
# currently, landmarks for the jaw region have been removed
def lmd_metric(landmark1, landmark2):
    lmd = np.linalg.norm(landmark1 - landmark2)
    return lmd / len(landmark1)

# face landmark distance metric
# compute the lmd without the jaw region
def landmark_distance_metric(source_image_path, target_image_path):
    # compute the average Euclidean distance between the two landmarks 

    source_landmark_path = source_image_path.replace('.png', '_landmarks.npz')
    target_landmark_path = target_image_path.replace('.png', '_landmarks.npz')

    source_image = cv2.cvtColor(cv2.imread(source_image_path), cv2.COLOR_BGR2RGB)
    target_image = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB)
    
    source_landmarks = np.load(source_landmark_path, allow_pickle=True)['landmark']
    target_landmarks = np.load(target_landmark_path, allow_pickle=True)['landmark']

    # first align the target image wrt to the source image using the landmarks     
    target_landmarks_transformed = landmark_alignment(source_image, 
                                    source_landmarks, target_image, target_landmarks)

    # landmark metric for the current pair of frames
    landmark_metric = lmd_metric(source_landmarks[17:], target_landmarks_transformed[17:])

    return landmark_metric

def landmark_distance_metric_video(pred_dir, gt_dir):
    current_pred_frames = sorted(glob(pred_dir + '/*.png'))
    current_gt_frames = sorted(glob(gt_dir + '/*.png'))

    total_landmark_distance = list()
    for index, (pred_frame, gt_frame) in enumerate(zip(current_pred_frames, current_gt_frames)):
        # compute the lmd metric between two pair of landmarks
        lmd = landmark_distance_metric(gt_frame, pred_frame)

        if math.isnan(lmd) or math.isinf(lmd):
            continue
        
        total_landmark_distance.append(lmd)

    return sum(total_landmark_distance) / len(total_landmark_distance)

def identity_matching_metric_video(pred_dir, gt_dir):
    current_pred_frames = sorted(glob(pred_dir + '/*.png'))
    current_gt_frames = sorted(glob(gt_dir + '/*.png'))

    total_id_metric = list()
    for index, (pred_frame, gt_frame) in enumerate(zip(current_pred_frames, current_gt_frames)):
        # compute the lmd metric between two pair of landmarks
        gt_embedding = dlib_feature_embedding(gt_frame)
        pred_embedding = dlib_feature_embedding(pred_frame)

        if gt_embedding is None or pred_embedding is None:
            continue

        identity_matching_metric = dlib_euclidean_distance(gt_embedding, pred_embedding)

        if math.isnan(identity_matching_metric) or math.isinf(identity_matching_metric):
            continue
        
        total_id_metric.append(identity_matching_metric)

    return sum(total_id_metric) / len(total_id_metric)

# assumptions - directory structure for videos
# each dir has results (predictions) about that baseline
# generate frames inside that baseline 
# a dir will be given as input that will have prediction videos
# generate the frames for the predictions videos in that dir
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # pred_dir is basically the baseline
    parser.add_argument('--gt_dir', type=str, default=None, help='ground truth directory')
    parser.add_argument('--pred_dir', type=str, default=None, help='prediction directory')
    parser.add_argument('--gt_generate_frames', action='store_true', help='generate gt frames or not')
    parser.add_argument('--pred_generate_frames', action='store_true', help='generate pred frames or not')
    parser.add_argument('--gt_comparison', default='source', help='default frames to compare to')
    args = parser.parse_args()

    gt_comparison = args.gt_comparison

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir

    gt_generate_frames = args.gt_generate_frames
    pred_generate_frames = args.pred_generate_frames

    if gt_generate_frames:
        print(f'generating gt frames')
        gt_videos = glob(gt_dir + '/*.mp4')
        for gt_video in gt_videos:
            save_dir_path = osp.join(gt_dir, osp.basename(gt_video).split('.')[0])
            os.makedirs(save_dir_path, exist_ok=True)

            frames = read_frames(gt_video)

            save_frames(frames, save_dir_path)

    if pred_generate_frames:
        print(f'generating pred frames')
        pred_videos = glob(pred_dir + '/*.mp4')
        for pred_video in pred_videos:
            save_dir_path = osp.join(pred_dir, osp.basename(pred_video).split('.')[0])
            os.makedirs(save_dir_path, exist_ok=True)

            frames = read_frames(pred_video)

            save_frames(frames, save_dir_path)

    # # the gt_dir and pred_dir are given as inputs
    # # both the dirs have the frames corresponding to videos inside their set 
    pred_dirs = list(set(glob(pred_dir + '/*')) - set(glob(pred_dir + '/*.mp4')))
    # gt_dirs = list(set(glob(gt_dir + '/*')) - set(glob(gt_dir + '/*.mp4')))
    print(f'Total pred dirs : {len(pred_dirs)}')

    # total_psnr = 0
    # total_ssim = 0

    # # for the pred_dirs find the corresponding gt_dir
    # for index, pred_dir in tqdm(enumerate(pred_dirs)):
    #     # find the corresponding gt dir 
    #     basename = '_'.join(osp.basename(pred_dir).split('_')[:3])
    #     gt_dir_path = osp.join(gt_dir, basename + '_' + gt_comparison)      

    #     # compute the ssim metric for the current video/(image dir)
    #     video_ssim = compute_ssim_video(pred_dir, gt_dir_path)
    #     print(f'video : {basename}, video_ssim : {video_ssim}')
        
    #     total_ssim += video_ssim

    # print('Mean ssim %.2f' % (total_ssim / len(pred_dirs)))

    total_lmd = 0

    # for the pred_dirs find the corresponding gt_dir
    for index, pred_dir in tqdm(enumerate(pred_dirs)):
        # find the corresponding gt dir 
        basename = '_'.join(osp.basename(pred_dir).split('_')[:3])
        gt_dir_path = osp.join(gt_dir, basename + '_' + gt_comparison)      

        # compute the ssim metric for the current video/(image dir)
        video_lmd = landmark_distance_metric_video(pred_dir, gt_dir_path)
        print(f'video : {basename}, video_lmd : {video_lmd}')
        
        total_lmd += video_lmd

    print('Mean lmd %.2f' % (total_lmd / len(pred_dirs)))


    # total_id_match = 0

    # # for the pred_dirs find the corresponding gt_dir
    # for index, pred_dir in tqdm(enumerate(pred_dirs)):
    #     # find the corresponding gt dir 
    #     basename = '_'.join(osp.basename(pred_dir).split('_')[:3])
    #     gt_dir_path = osp.join(gt_dir, basename + '_' + gt_comparison)      

    #     # compute the ssim metric for the current video/(image dir)
    #     video_id_match = identity_matching_metric_video(pred_dir, gt_dir_path)
    #     print(f'video : {basename}, video_id_match : {video_id_match}')
        
    #     total_id_match += video_id_match

    # print('Mean id match %.2f' % (total_id_match / len(pred_dirs)))