from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd


def init_ff(phase,level='frame',n_frames=8):
	#dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'
	if phase=='train':
		dataset_path='../frames/train/'
	elif phase=='val':
		dataset_path='../frames/val/'
	else:
		dataset_path='../frames/test/'
	image_list=[]
	label_list=[]

	
	#manual dataset split
	#split the data like /train, /val,/test
	folder_list = sorted(glob(dataset_path+'*'))
	'''
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	'''
	
	
	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		#if n_frames<len(images_temp):
			#images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)
	return image_list,label_list


