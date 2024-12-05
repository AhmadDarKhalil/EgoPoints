# EgoPoints

# Evaluation
## PIPs++
Note: The PIPs++ evaluation script expects a folder path for the `--ckpt_path` argument. Their model loader function will then look in this folder for the most recent training step file. For ease of use, simply create a folder for each model you wish to evaluate.
```
cd EgoPoints
git clone https://github.com/aharley/pips2.git
cd pips2
git checkout 8b5bd9ecb27274f76f75fcaeff0dbdf13de0b977
conda create -n pips2 python=3.8
conda activate pips2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd ..
python3 egopoints_eval_pips2.py --dataset_location=/path/to/ego_points/folder --ckpt_path=/folder/containing/checkpoint 
```

## CoTracker2
Note: Our CoTracker2 evaluation script uses util functions from the pips2 repository, so you will need to do at least the first 3 lines from the PIPs++ section to run this evaluation.
```
cd EgoPoints
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
git checkout 9921cf0895b5eccf57666f3652e298a04646bcd3
conda create -n cotracker_env python=3.8
conda activate cotracker_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
pip install opencv-python imageio scikit-learn scikit-image prettytable
cd ..
python3 egopoints_eval_cotracker.py --dataset_location=/path/to/ego_points/folder --ckpt_path=/path/to/checkpoint/file
```
