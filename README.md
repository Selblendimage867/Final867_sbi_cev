# Application of self blended image in Cross-efficient-vit

Merging the two exsiting projects together and optimization on exisiting code<br>
Code references:
Self-blend-iamge : https://github.com/mapooon/selfblendedimages <br>
Cross_efficient-vit: https://github.com/Selblendimage867/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection <br>
I will only introduce the sbi implementation here, for the use of vanilla cross-efficient-nent, refer to the link above.<br>
The template is also copied from Cross-efficient-vit.<br>
Another word, if you are not familiar with Deepfakes, the code might turn out be buggy to you.<br>
Sometimes you should try to tune your environment or simply ask google<br>
Most of the neural network implementation will behave odd on other machines, it is not uncommon, I referenced from published papers and the code still have bugs.<br>
Stay calm and raise any questions you encounter.<br>
# Setup
Clone the repository and move into it:
```
git clone https://github.com/Selblendimage867/Final867_sbi_cev.git

cd Final867_sbi_cev
```

Setup Python environment using conda:<br>
For SBI training
```
conda env create --file environment.yml
conda activate sbi
export PYTHONPATH=.
```

# Get the data
Download and extract the dataset you want to use from:
- FaceForensics++: https://github.com/ondyari/FaceForensics/blob/master/dataset/
The network is trianed on c23 compression be careful with your download options.


# Preprocess the data
During preprocessing, there might be cases that you have encountered bugs with directories, I have make a comment in the get_video_path() function in preprocess/utils.py<br>
Try to play with the comments and your program will run without bugs.<br>
Don't worry, this will not affect the training.<br>

```
cd preprocessing
python3 detect_faces_sbi.py --data_path "path/to/videos" --dataset FACEFORENSICS
```

You will see a /boxes folder at the Final867_sbi_cev directory.<br>

#Facecropping
For SBI implementation:
```
python3 crop_dlib_ff.py --d Original -c23

```
Check out the crop_dlib_ff.py for arguments, -c stands for compresssion<br>
<br>
Below the if __name__=="main": section<br>
you will see a lot of path, try to mainpulate this line to the correct directory you set.<br>
if args.dataset=='Original':<br>
        dataset_path='../your_folder/original_sequences/youtube/{}/'.format(args.comp)<br>

After you have done all the steps, do this for boxes, frames and landmark folder 
```
- boxes or frames or landmark
    - train
            - video_name_0
                0_0.png
                1_0.png
                2_0.png
                ...
                N_0.png
            ...
            - video_name_K
                0_0.png
                1_0.png
                2_0.png
                ...
                M_0.png
        
    - val
        ...
            ...
        ...
            ...
    - test
        ...
            ...
        ...
            ...

The vanila implementaitons also requires you to manually do the test-train data split, don't blame me for not making your life easier, even top researchers don't pay attention to this.<br> 

# Evaluate
Move into the choosen architecture folder you want to evaluate and download the pre-trained model:<br>
We didn't change the efficient-vit implementation so i don't recommend you to do that

```
cd cross-efficient-vit
```
We will post our pretrained models soon, but you are recommended to train them yourselves


Then, issue the following commands for evaluating a given model giving the pre-trained model path and the configuration file available in the config directory:
```
python3 test_sbi.py --model_path "pretrained_models/[model]" --config "configs/architecture.yaml"
```

By default the command will test on DFDC dataset but you can customize the following parameters for both the architectures:
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)
- --max_videos: Maximum number of videos to use for training (default: all)
- --workers: Number of data loader workers (default: 10)
- --frames_per_video: Number of equidistant frames for each video (default: 30)
- --batch_size: Prediction Batch Size (default: 32)

    
To evaluate a customized model trained from scratch with a different architecture you need to edit the configs/architecture.yaml file.

# Train

(Cross Efficient ViT)
```
cd cross-efficient-vit
python3 train_sbi.py --config configs/architecture.yaml
```

By default the commands will train on DFDC dataset but you can customize the following parameters for both the architectures:
- --num_epochs: Number of training epochs (default: 300)
- --workers: Number of data loader workers (default: 10)
- --resume: Path to latest checkpoint (default: none)
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All) (default: All)
- --max_videos: Maximum number of videos to use for training (default: all)
- --patience: How many epochs wait before stopping for validation loss not improving (default: 5)
    

Only for the Efficient ViT model it's also possible to custom the patch extractor and use different versions of EfficientNet (only B0 and B7) by adding the following parameter:
- --efficient_net: Which EfficientNet version to use (0 or 7, default: 0)


