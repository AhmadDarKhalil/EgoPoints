# EgoPoints

# Evaluation
## PIPs++
```
cd EgoPoints
git clone https://github.com/aharley/pips2.git
cd pips2
git checkout acf40f0af4019ad5e570ebbc210286c0720f2fab 
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
pip install opencv-python
pip install imageio
cd ..
python3 egopoints_eval_cotracker.py --dataset_location=/path/to/ego_points/folder --ckpt_path=/path/to/checkpoint/file
```
