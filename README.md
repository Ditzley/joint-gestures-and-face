# Joint Co-Speech Gesture and Expressive Talking Face Generation using Diffusion with Adapters [WACV2025]

The official PyTorch implementation of the **WACV2025** paper [**"Joint Co-Speech Gesture and Expressive Talking Face Generation using Diffusion with Adapters"**](https://arxiv.org/abs/2412.14333).

Please visit our [**webpage**](https://Ditzley.github.io/joint-gestures-and-face) for more details.


## Getting started

This code was tested on `Ubuntu 20.04.6 LTS`.

### 1. Setup environment

Clone the repo:
  ```bash
  git clone https://github.com/Ditzley/joint-gestures-and-face
  cd joint-gestures-and-face
  ```  
Create conda environment:
```bash
conda create --name joint python=3.11
conda activate joint
```
Install pytorch:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
    
Install other requirements:

```bash
pip install lightning smplx trimesh opencv-python timm einops transformers lmdb jsonargparse[signatures]>=4.27.7
```

Download SMPLX model at https://smpl-x.is.tue.mpg.de/. Place it in ``<path-to-repo>/visualise/smplx_model``. Also download the extra files from https://github.com/yhw-yhw/TalkSHOW/tree/main/visualise/smplx and place in ``<path-to-repo>/visualise/smplx_model``.


### 2. Get data

Please follow [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW) for downloading and preparing the dataset.

### 4. Training

To train the model run:
```bash
python scripts/run.py --config config/diffusion.json --train --exp_name <experiment>
```

### 5. Testing

To test the model run:
```bash
python scripts/run.py --config config/diffusion.json --model_path <model_path> --infer
```

### 5. Visualization

Our prediction code outputs 2 .npy files, one containing the joints and expressions, and another containing the smplx vertices. You can use the visualisation code from [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW) to render videos. 

## Citation
If you find our work useful to your research, please consider citing:
```
@misc{hogue2024jointcospeechgestureexpressive,
      title={Joint Co-Speech Gesture and Expressive Talking Face Generation using Diffusion with Adapters}, 
      author={Steven Hogue and Chenxu Zhang and Yapeng Tian and Xiaohu Guo},
      year={2024},
      eprint={2412.14333},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2412.14333}, 
}
```

## Acknowledgements
We thank the following authors for their work:
 - [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW) for the SHOW dataset and their data scripts which we use and base ours on
 - [DiffGesture](https://github.com/Advocate99/DiffGesture/) on which we base our diffusion and transformer models
 - [LAVisH](https://github.com/GenjiB/LAVISH) on which we base our adapter module

