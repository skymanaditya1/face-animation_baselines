# This is the code for preprocessing the data and running the different baselines 
import os
import os.path as osp
import random
import subprocess
from glob import glob
from tqdm import tqdm

import cv2

fomm_working_dir = '/ssd_scratch/cvit/aditya1/baselines/fomm/'
cosegmentation_working_dir = '/ssd_scratch/cvit/aditya1/baselines/motion-cosegmentation/'

baseline_results_dir = '/ssd_scratch/cvit/aditya1/baselines/results'
temp_dir = '/ssd_scratch/cvit/aditya1/baselines/temp'

base_dir = '/ssd_scratch/cvit/aditya1/baselines/custom_data'
good_video_files = osp.join(base_dir, 'good_files')
bad_video_files = osp.join(base_dir, 'bad_files')

def read_frames(video_path):
	video_stream = cv2.VideoCapture(video_path)
	ret, frame = video_stream.read()
	frames = list()

	while ret:
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		ret, frame = video_stream.read()

	return frames

def save_image_frame(frame, path):
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imwrite(path, frame)

def save_image_frames(frames, save_dir):
	for index, frame in enumerate(frames):
		save_path = osp.join(save_dir, str(index).zfill(3) + '.png')
		save_image_frame(frame, save_path)


def save_frames_as_video(frames, video_path, fps=25):
	height, width, layers = frames[0].shape

	video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
	for frame in frames: 
		# video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
		video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	  
	cv2.destroyAllWindows() 
	video.release()



# run the fomm baseline
# fomm needs an image that is animated using the target video
def run_fomm():
	good_videos = glob(good_video_files + '/*_source.mp4')
	random_idx = random.randint(0, len(good_videos)-1)
	random_source_video = good_videos[random_idx]

	print(f'source video : {random_source_video}')

	target_video = random_source_video.replace('source', 'target')

	# sample a random frame from the target video 
	target_frames = read_frames(target_video)
	target_idx = random.randint(0, len(target_frames)-1)
	random_target_frame = target_frames[target_idx]

	fomm_temp_dir = osp.join(temp_dir, 'fomm_temp')
	os.makedirs(fomm_temp_dir, exist_ok=True)

	target_frame_path = osp.join(fomm_temp_dir, osp.basename(random_source_video).split('.')[0] + '_' + str(target_idx)) + '.png'
	print(f'target frame path : {target_frame_path}')

	save_image_frame(random_target_frame, target_frame_path)

	fomm_results_dir = osp.join(baseline_results_dir, 'fomm_results')
	os.makedirs(fomm_results_dir, exist_ok=True)

	result_video = osp.join(fomm_results_dir, osp.basename(random_source_video).split('.')[0] + '_target_frame_' + str(target_idx).zfill(3) + '.mp4')
	print(f'result video : {result_video}')

	# run the fomm baseline using the pretrained model 
	# random_source_video as the driving video and random_target_frame as the static image frame
	dataset_name = 'config/vox-256.yaml'
	checkpoint = 'checkpoints/vox-cpk.pth.tar'

	fomm_template = 'python demo.py  --config {} --driving_video {} \
				 --source_image {} --checkpoint {} --relative --adapt_scale \
				 --result_video {}'

	fomm_command = fomm_template.format(dataset_name, random_source_video, 
					target_frame_path, checkpoint, result_video)

	print(f'fomm_command : {fomm_command}')

	os.chdir(fomm_working_dir)
	os.system(fomm_command)


# runs the fomm baseline by animating a source image using a single frame of target video
def run_fomm_video(src_video=None, tgt_video=None):
	if src_video is None:
		good_videos = glob(good_video_files + '/*_source.mp4')
		random_idx = random.randint(0, len(good_videos)-1)
		random_source_video = good_videos[random_idx]

	else:
		random_source_video = src_video

	print(f'source video : {random_source_video}')

	if tgt_video is None:
		target_video = random_source_video.replace('source', 'target')
	else:
		target_video = tgt_video
	
	print(f'target video : {target_video}')

	# generate frames from the target video 
	# the target frame needs to be animated using the source video 
	target_frames = read_frames(target_video)

	# target_idx = random.randint(0, len(target_frames)-1)
	# random_target_frame = target_frames[target_idx]

	def get_random(random_chars=5):
		chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
		return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(random_chars)])

	intermediate_dir = 'fomm_video_temp'
	video_results = 'fomm_video_results'

	target_frames_dir = osp.join(osp.join(temp_dir, intermediate_dir), get_random())
	os.makedirs(target_frames_dir, exist_ok=True)

	save_image_frames(target_frames, target_frames_dir)
	print(f'target frame dir path : {target_frames_dir}')

	# saves videos of 1 frame duration each
	temp_video_dir = osp.join(osp.join(temp_dir, intermediate_dir), get_random())
	os.makedirs(temp_video_dir, exist_ok=True)

	print(f'temp video dir (1s) duration videos : {temp_video_dir}')

	# to run the below command, we need to create videos of duration 1 frame each 
	source_frames = read_frames(random_source_video)
	for index, frame in enumerate(source_frames):
		video_path = osp.join(temp_video_dir, str(index).zfill(3) + '.mp4')
		save_frames_as_video([frame], video_path)

	temp_results_dir = osp.join(osp.join(temp_dir, intermediate_dir), get_random())
	os.makedirs(temp_results_dir, exist_ok=True)

	# there are three possible configurations - 5 segments and 10 segments

	# run the fomm baseline using the pretrained model 
	# random_source_video as the driving video and random_target_frame as the static image frame
	dataset_name = 'config/vox-256.yaml'
	checkpoint = 'checkpoints/vox-cpk.pth.tar'

	template = 'python demo.py  --config {} --driving_video {} \
				 --source_image {} --checkpoint {} --relative --adapt_scale \
				 --result_video {}'


	# run the cosegmentation baseline using the pretrained model on all the videos
	source_videos = sorted(glob(temp_video_dir + '/*.mp4'))
	target_frames = sorted(glob(target_frames_dir + '/*.png'))

	os.chdir(fomm_working_dir)

	for index, source_video in tqdm(enumerate(source_videos)):
		result_video = osp.join(temp_results_dir, str(index).zfill(3) + '.mp4')

		command = template.format(dataset_name, source_video, 
					target_frames[index], checkpoint, result_video)
		
		os.system(command)

	print(f'temporary swapped results : {temp_results_dir}')

	# read the frames from the saved videos, and combine the frames to generate the resulting video
	temp_result_videos = sorted(glob(temp_results_dir + '/*.mp4'))
	temp_result_frames = list()
	for temp_result_video in temp_result_videos:
		temp_result_frames.extend(read_frames(temp_result_video))


	video_results_dir = osp.join(baseline_results_dir, video_results)
	os.makedirs(video_results_dir, exist_ok=True)

	result_video = osp.join(video_results_dir, 
					osp.basename(random_source_video).split('.')[0] + '.mp4')

	print(f'result video : {result_video}')

	# save the result video
	save_frames_as_video(temp_result_frames, result_video)

# run the pcavs baseline 
def run_pcavs():
	pass

# run the makeittalk baseline 


# runs motion cosegmentation swap using only a single frame from the input
# -- flaw - loss in identity information 
def run_cosegmentation_swap_frame():
	good_videos = glob(good_video_files + '/*_source.mp4')
	random_idx = random.randint(0, len(good_videos)-1)
	random_source_video = good_videos[random_idx]

	print(f'source video : {random_source_video}')

	target_video = random_source_video.replace('source', 'target')

	# sample a random frame from the target video 
	target_frames = read_frames(target_video)
	target_idx = random.randint(0, len(target_frames)-1)
	random_target_frame = target_frames[target_idx]

	fomm_temp_dir = osp.join(temp_dir, 'cosegmentation_frame_temp')
	os.makedirs(fomm_temp_dir, exist_ok=True)

	target_frame_path = osp.join(fomm_temp_dir, osp.basename(random_source_video).split('.')[0] + '_' + str(target_idx)) + '.png'
	print(f'target frame path : {target_frame_path}')

	save_image_frame(random_target_frame, target_frame_path)

	cosegmentation_results_dir = osp.join(baseline_results_dir, 'cosegmentation_frame_results')
	os.makedirs(cosegmentation_results_dir, exist_ok=True)

	result_video = osp.join(cosegmentation_results_dir, osp.basename(random_source_video).split('.')[0] + '_target_frame_' + str(target_idx).zfill(3) + '.mp4')
	print(f'result video : {result_video}')

	# run the fomm baseline using the pretrained model 
	# random_source_video as the driving video and random_target_frame as the static image frame

	# there are three possible configurations - 5 segments and 10 segments

	# 10 segment code
	dataset_name = 'config/vox-256-sem-10segments.yaml'
	checkpoint = 'checkpoints/vox-10segments.pth.tar'
	swap_indices = '2,5,7'

	# # 5 segment code 
	# dataset_name = 'config/vox-256-sem-5segments.yaml'
	# checkpoint = 'checkpoints/vox-5segments.pth.tar'
	# swap_indices = '2'

	template = 'python part_swap.py --config {} --target_video {} \
							--source_image {} --checkpoint {} --swap_index {} \
							--result_video {}'


	command = template.format(dataset_name, random_source_video, 
					target_frame_path, checkpoint, swap_indices, 
					result_video)

	print(f'cosegmentation frame command : {command}')

	os.chdir(cosegmentation_working_dir)
	os.system(command)


# runs motion cosegmentation swap using videos as both swap and animation
def run_cosegmentation_swap_video(src_file=None, dst_file=None):

	if src_file is None:
		good_videos = glob(good_video_files + '/*_source.mp4')
		random_idx = random.randint(0, len(good_videos)-1)
		random_source_video = good_videos[random_idx]
	else:
		random_source_video = src_file

	print(f'source video : {random_source_video}')

	if dst_file is None:
		target_video = random_source_video.replace('source', 'target')
	else:
		target_video = dst_file

	print(f'target video : {target_video}')

	# sample a random frame from the target video 
	target_frames = read_frames(target_video)

	# target_idx = random.randint(0, len(target_frames)-1)
	# random_target_frame = target_frames[target_idx]

	def get_random(random_chars=5):
		chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
		return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(random_chars)])

	target_frames_dir = osp.join(osp.join(temp_dir, 'cosegmentation_video_temp'), get_random())
	os.makedirs(target_frames_dir, exist_ok=True)

	save_image_frames(target_frames, target_frames_dir)
	print(f'target frame dir path : {target_frames_dir}')

	temp_video_dir = osp.join(osp.join(temp_dir, 'cosegmentation_video_temp'), get_random())
	os.makedirs(temp_video_dir, exist_ok=True)

	print(f'temp video dir (1s) duration videos : {temp_video_dir}')

	# to run the below command, we need to create videos of duration 1 frame each 
	source_frames = read_frames(random_source_video)
	for index, frame in enumerate(source_frames):
		video_path = osp.join(temp_video_dir, str(index).zfill(3) + '.mp4')
		save_frames_as_video([frame], video_path)

	cosegmentation_temp_results_dir = osp.join(osp.join(temp_dir, 'cosegmentation_video_temp'), get_random())
	os.makedirs(cosegmentation_temp_results_dir, exist_ok=True)

	# there are three possible configurations - 5 segments and 10 segments

	# 10 segment code
	dataset_name = 'config/vox-256-sem-10segments.yaml'
	checkpoint = 'checkpoints/vox-10segments.pth.tar'
	swap_indices = '2,5,7'

	# # 5 segment code 
	# dataset_name = 'config/vox-256-sem-5segments.yaml'
	# checkpoint = 'checkpoints/vox-5segments.pth.tar'
	# swap_indices = '2'

	template = 'python part_swap.py --config {} --target_video {} \
							--source_image {} --checkpoint {} --swap_index {} \
							--result_video {}'


	# run the cosegmentation baseline using the pretrained model on all the videos
	source_videos = sorted(glob(temp_video_dir + '/*.mp4'))
	target_frames = sorted(glob(target_frames_dir + '/*.png'))

	os.chdir(cosegmentation_working_dir)

	for index, source_video in tqdm(enumerate(source_videos)):
		result_video = osp.join(cosegmentation_temp_results_dir, str(index).zfill(3) + '.mp4')
		command = template.format(dataset_name, source_video, 
					target_frames[index], checkpoint, swap_indices,
					result_video)
		
		os.system(command)

	print(f'temporary swapped results : {cosegmentation_temp_results_dir}')

	# read the frames from the saved videos, and combine the frames to generate the resulting video
	temp_result_videos = sorted(glob(cosegmentation_temp_results_dir + '/*.mp4'))
	temp_result_frames = list()
	for temp_result_video in temp_result_videos:
		temp_result_frames.extend(read_frames(temp_result_video))


	cosegmentation_video_results_dir = osp.join(baseline_results_dir, 'cosegmentation_video_results')
	os.makedirs(cosegmentation_video_results_dir, exist_ok=True)

	result_video = osp.join(cosegmentation_video_results_dir, 
					osp.basename(random_source_video).split('.')[0] + '_' + get_random() + '.mp4')

	print(f'result video : {result_video}')

	# save the result video
	save_frames_as_video(temp_result_frames, result_video)


# run the deepfakes baseline 


# run the deepfacelabs baseline 
# command for running one shot neural view synthesis 
'''python demo.py --config config/vox-256.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_image /ssd_scratch/cvit/aditya1/baselines/custom_validation/fomm/example5_bigguy/dst.png --driving_video /ssd_scratch/cvit/aditya1/baselines/custom_validation/fomm/example5_bigguy/src.mp4 --relative --adapt_scale --result_video /home2/aditya1/cvit/content_sync/acm_results/custom_validation/fomm/example5_bigguy/fomm.mp4
'''


if __name__ == '__main__':

	baselines = ['fomm', 'pcavs', 'makeittalk', 
				'cosegmentation_frame', 'cosegmentation_video', 
				'fomm_video', 'deepfakes', 'deepfacelabs']
	baseline = baselines[4]
                           
	print(f'Running baseline {baseline}')

	# if baseline == 'fomm':
	# 	run_fomm()
	# elif baseline == 'cosegmentation_frame':
	# 	run_cosegmentation_swap_frame()
	# elif baseline == 'cosegmentation_video':
	# 	run_cosegmentation_swap_video()
	# elif baseline == 'fomm_video':
	# 	run_fomm_video()

	# src_video = '/ssd_scratch/cvit/aditya1/baselines/face-animation_baselines/custom_example/source_video.mp4'
	# tgt_video = '/ssd_scratch/cvit/aditya1/baselines/face-animation_baselines/custom_example/target_video.mp4'

	# test1/1_0_113, test2/1_0_18, test3/1_0_26, test4/1_0_5, test5/1_0_63

	src_video = '/ssd_scratch/cvit/aditya1/baselines/custom_validation/cosegmentation/test5/1_0_63_dst.mp4'
	tgt_video = '/ssd_scratch/cvit/aditya1/baselines/custom_validation/cosegmentation/test5/1_0_63_src.mp4'

	run_cosegmentation_swap_video(src_video, tgt_video)