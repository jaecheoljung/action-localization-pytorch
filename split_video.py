from __future__ import print_function, division
import os
import sys
import subprocess
import random
import glob

num_people = 20
num_val = 4
mylist = random.sample(range(num_people), num_val)

if __name__=="__main__":
  root_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  dst_train_dir = os.path.join(dst_dir_path, 'train')
  dst_val_dir = os.path.join(dst_dir_path, 'val')
  os.mkdir(dst_dir_path)
  os.mkdir(dst_train_dir)
  os.mkdir(dst_val_dir)

  for label in sorted(os.listdir(root_path)):
  	path = os.path.join(root_path, label)
  	videos = sorted(os.listdir(path))
  	for video in videos:
  		if videos.index(video) not in mylist:
  			dst = os.path.join(dst_train_dir, label)
  		else:
  			dst = os.path.join(dst_val_dir, label)
  		if not os.path.exists(dst):
  			os.mkdir(dst)
  		name, ext = os.path.splitext(video)
  		original_video_file = os.path.join(path, video)
  		new_video_file = os.path.join(dst, name)
  		cmd = 'ffmpeg -i \"{}\" -acodec copy -f segment -segment_time 8 -vcodec copy -reset_timestamps 1 -map 0 \"{}\"c%03d.avi'.format(original_video_file, new_video_file)
  		print(cmd)
  		subprocess.call(cmd, shell=True)
  		print('\n')
  		os.remove(os.path.join(dst, os.listdir(dst)[-1]))
  		os.remove(glob.glob(os.path.join(dst, '*c000.avi'))[0])