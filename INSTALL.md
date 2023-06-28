## Installation

### Requirements

Requirements are the same as [Mask2Former](https://github.com/facebookresearch/Mask2Former).

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

```bash
cd caroq/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup
```bash
conda create --name caroq python=3.8 -y
conda activate caroq
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# clone repository
git clone https://github.com/AnwesaChoudhuri/CAROQ.git
cd CAROQ
pip install -r requirements.txt
cd caroq/modeling/pixel_decoder/ops
sh make.sh
```
