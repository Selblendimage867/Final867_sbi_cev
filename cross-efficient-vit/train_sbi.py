import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
#from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
#from utils.util import get_method, check_correct, resize, shuffle_dataset, get_n_params
from utils.util_train import check_correct, get_n_params
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
from utils.sbi import SBI_Dataset
import collections
#from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from utils.logs import log
import matplotlib.pyplot as plt
BASE_DIR = '../'
DATA_DIR = os.path.join(BASE_DIR, "frames")
TRAINING_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_PATH = "models_sbi"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")
'''
def read_frames(video_path, train_dataset, validation_dataset):
    
    # Get the video label based on dataset selected
    method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                video_folder_name = os.path.basename(video_path)
                video_key = video_folder_name + ".mp4"
                if video_key in metadata.keys():
                    item = metadata[video_key]
                    label = item.get("label", None)
                    if label == "FAKE":
                        label = 1.         
                    else:
                        label = 0.
                    break
                else:
                    label = None
        else:
            label = 1.
        if label == None:
            print("NOT FOUND", video_path)
    else:
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
        else:
            label = 1.

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)

    
    
    if VALIDATION_DIR in video_path:
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))
'''
# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    #parser.add_argument('--dataset', type=str, default='All', 
                        #help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)
    #make log file
    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    logger = log(path="./logs/", file="sbi.logs")
    #load configuration
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    device = torch.device('cuda')
    #our model is the cross-efficient-vit model
    model = CrossEfficientViT(config=config)
    model.train()   
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
   
    #READ DATASET
    #if opt.dataset != "All":
        #folders = ["Original", opt.dataset]
    #else:
        #folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    '''
    sets = [TRAINING_DIR, VALIDATION_DIR]
    folders = ["Original"]
    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                if index == opt.max_videos:
                    break
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))
    '''
    '''         
    #changes to data reading method
    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()
    #show training progress
    with Pool(processes=10) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset),paths):
                pbar.update()
    #original frames self blend to fake frames
    #so number double up
    
    train_samples = len(train_dataset)*2
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)*2
    validation_dataset = shuffle_dataset(validation_dataset)
    '''
    #changes done to reading method
    # Print some useful statistics
    #print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    #print("__TRAINING STATS__")
    #train_counters = collections.Counter(image[1] for image in train_dataset)
    #print(train_counters)
    class_weights = 1
    '''
    #class_weights = train_counters[0] / train_counters[1]
    
    print("Weights", 1)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")
    '''
    #pos_weight=torch.tensor([class_weights]).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    #loss_fn = torch.nn.CrossEntropyLoss()
   
    
    # Create the data loaders
    #validation_labels = np.asarray([row[1] for row in validation_dataset])
    #labels = np.asarray([row[1] for row in train_dataset])

    #train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), labels, config['model']['image-size'])
    train_dataset = SBI_Dataset(phase='train',image_size= config['model']['image-size'])
    '''
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    '''
    
    train_sample = int(train_dataset.__len__())
    print(train_sample)
    #validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
    val_dataset = SBI_Dataset(phase='val',image_size= config['model']['image-size'])
    '''
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    '''
    val_sample =int(val_dataset.__len__())
    dl=torch.utils.data.DataLoader(train_dataset,
                        batch_size=config['training']['bs'],
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn)
    #del train_dataset
    val_dl=torch.utils.data.DataLoader(val_dataset,
                        batch_size=config['training']['bs'],
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    #del val_dataset
    
    
    
    

    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf

    train_loss_log = np.empty((0,0))
    train_acc_log = np.empty((0,0))
    val_loss_log = np.empty((0,0))
    val_acc_log = np.empty((0,0))
    #store loss and accurcy trial for image generation
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
       
        train_correct = 0
        positive = 0
        negative = 0
        for index, data in enumerate(tqdm(dl)):
            
            images=data['img'].float()
            
            target=data['label'].to(device, non_blocking=True).long()
            
            images = np.transpose(images, (0, 1, 3, 2))
            
            images = images.cuda()
            y_pred = model(images)
            y_pred = torch.squeeze(y_pred, 1)
            
            

            
            #print('compare pred',y_pred, "and target",target)
            loss = loss_fn(y_pred, target.float())
        
            corrects, positive_class, negative_class = check_correct(y_pred, target)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(config['training']['bs']):
                bar.next()

             
            if index%1200 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(2*counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)  
        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
                        t+1,
                        opt.num_epochs,
                        total_loss/counter,
                        train_correct/(2*counter*config['training']['bs']),
                        )
        train_loss_log=np.append(train_loss_log,float(total_loss/counter))
        train_acc_log=np.append(train_acc_log,float(train_correct/(2*counter*config['training']['bs'])))
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
       
        train_correct /= (2*train_sample) #number of samples at test
        total_loss /= counter
        
        for index, data in enumerate(tqdm(val_dl)):
            
            val_images=data['img'].float()
            
            val_target=data['label'].to(device, non_blocking=True).long()
            val_images = np.transpose(val_images, (0, 1, 3, 2))
            
            val_images = val_images.cuda()
            val_pred = model(val_images)
            val_pred = torch.squeeze(val_pred, 1)
            val_loss = loss_fn(val_pred, val_target.float())
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_target)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            bar.next()
            
        scheduler.step()
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= (2*val_sample) #number of samples at validation
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + " val_1s:" + str(val_positive))
        log_text+="val loss: {:.4f}, val acc: {:.4f}".format(
                        total_val_loss,
                        val_correct
                        )
        val_loss_log=np.append(val_loss_log,float(total_val_loss))
        val_acc_log=np.append(train_acc_log,float(val_correct))
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        if t%5 == 0:
            torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnet_checkpoint" + str(t) + "_" + 'sbi'))
            
        logger.info(log_text)
        #the _log numpy arrays is used for recording accuracy and loss in train and validation loss, if you would like
        # you may write something down here to extract them and plot a graph  

        
        
