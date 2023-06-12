## Installation

### Requirements

Requirements are the same as Mask2Former: https://github.com/facebookresearch/Mask2Former

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup
```bash
conda create --name clipsqueries python=3.8 -y
conda activate clipsqueries
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

# unzip Codes
cd CARQ_Prop
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
