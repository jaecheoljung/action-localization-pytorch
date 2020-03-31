from statistics import mean
import glob
import numpy as np
import cv2

if __name__ == "__main__":
	mean_rgb = [0] * 3
	imgs = [img for img in glob.glob('data/**', recursive = True) if img.endswith(".jpg")]
	for img in imgs:
		image = cv2.imread(img)
		rgb = image.mean(axis=0).mean(axis=0)
		for i in range(3):
			mean_rgb[i] += rgb[i]

	print([i/len(imgs) for i in mean_rgb])