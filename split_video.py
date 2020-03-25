from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, dst_dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  if not os.path.exists(dst_dir_path):
    os.mkdir(dst_dir_path)

  dst_class_path = os.path.join(dst_dir_path, class_name)
  if not os.path.exists(dst_class_path):
    os.mkdir(dst_class_path)

  for file_name in os.listdir(class_path):
    if '.avi' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    original_video_file = os.path.join(class_path, file_name)
    new_video_file = os.path.join(dst_class_path, name)

    cmd = 'ffmpeg -i \"{}\" -acodec copy -f segment -segment_time 8 -vcodec copy -reset_timestamps 1 -map 0 \"{}\"c%03d.avi'.format(original_video_file, new_video_file)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')
    os.remove(os.path.join(dst_class_path, os.listdir(dst_class_path)[-1]))

if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  for class_name in os.listdir(dir_path):
    class_process(dir_path, dst_dir_path, class_name)
