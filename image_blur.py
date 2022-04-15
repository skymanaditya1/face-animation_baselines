# script for adding a variety of image processing blurs on a sequence of frames 
from glob import glob 
import cv2
import os
import os.path as osp
import numpy as np
import random
import argparse

def apply_blurs(images, kernel_size=10):
	return [apply_blur(x, kernel_size) for x in images]

def apply_blur(image, kernel_size):
	blur_img = cv2.blur(image, (kernel_size, kernel_size), 0)
	return blur_img

def save_frame(frame, filepath):
    cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def save_frames(frames, save_dir_path):
	for index, frame in enumerate(frames):
		save_filepath = osp.join(save_dir_path, str(index).zfill(5) + '.png')
		save_frame(frame, save_filepath)

def apply_histograms(images):
	return [histogram_equalize(x) for x in images]

def histogram_equalize(img):
	img_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

	return hist_eq

def gamma_correction(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def adjust_gamma_frames(images):
	return [gamma_correction(image) for image in images]

def read_frame(filename):
	image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
	return image

def read_frames(dir_path):
	image_paths = sorted(glob(dir_path + '/*.png'))
	frames = list()
	for image_path in image_paths:
		frame = read_frame(image_path)
		frames.append(frame)

	return frames

def color_normalize_frame(img, image_size=256):
	normalizedImg = np.zeros((image_size, image_size))
	normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
	return normalizedImg

def color_normalize_frames(images):
	color_normalized_images = list()
	for index, image in enumerate(images):
		normalized_image = color_normalize_frame(image)
		color_normalized_images.append(normalized_image)

	return color_normalized_images


images = glob('frames/*.png')


def get_random(char_len=5):
	chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
	return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(char_len)])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, default=None, help='sample dir to run blur on')
	parser.add_argument('--kernel_size', type=int, default=5, help='kernel size')
	args = parser.parse_args()

	dir_path = args.input_dir
	frames = read_frames(dir_path)
	kernel_size = args.kernel_size

	color_blur, gaussian_blur, apply_histogram, adjust_gamma = True, True, False, False

	output_img_dir = osp.basename(dir_path) + '_blurred_k' + str(kernel_size) + '_' + get_random()
	os.makedirs(output_img_dir, exist_ok=True)

	if color_blur:
		frames = color_normalize_frames(frames)
	if apply_histogram:
		frames = apply_histograms(frames)
	if adjust_gamma:
		frames = adjust_gamma_frames(frames)
	if gaussian_blur:
		frames = apply_blurs(frames, kernel_size)

	print(f'Saving to output directory : {output_img_dir}')

	# save the blurred images to the disk 
	save_frames(frames, output_img_dir)