# Context-Aware Relative Object Queries to Unify Video Instance and Panoptic Segmentation (CVPR 2023)
by [Anwesa Choudhuri](https://ece.illinois.edu/about/directory/grad-students/anwesac2/), [Girish Chowdhary](https://ece.illinois.edu/about/directory/faculty/girishc/), and [Alexander G. Schwing](http://www.alexander-schwing.de/)


[[Project Page](https://anwesachoudhuri.github.io/ContextAwareRelativeObjectQueries/)] [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Choudhuri_Context-Aware_Relative_Object_Queries_To_Unify_Video_Instance_and_Panoptic_CVPR_2023_paper.pdf)] [[BibTeX](https://anwesachoudhuri.github.io/ContextAwareRelativeObjectQueries/bib.txt)]

We develop a simple approach for multiple video segmentation tasks: video instance segmentation, multi-object tracking and segmentation and video panoptic segmentation, using the propagation of context-aware relative object queries (CAROQ).


## Installation

See [INSTALL.md](https://github.com/AnwesaChoudhuri/CAROQ/blob/master/INSTALL.md).


## Getting Started

### Dataset preparation 
See [datasets/README.md](https://github.com/AnwesaChoudhuri/CAROQ/blob/master/datasets/README.md).

### Download Models

Please create a directory called **./models** under the home directory and place all initial models for training and trained models for evaluation.
For training, we start with [Mask2Former models](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).
Specific models used for initialization are mentioned in the config files for each dataset. Model paths can also be specified in the config files.

Trained models are coming soon!

### Training/Evaluation

See [train_eval_script.sh](https://github.com/AnwesaChoudhuri/CAROQ/blob/master/train_eval_script.sh).


## Citation

If you find the code or paper useful, please cite the following BibTeX entry.

```BibTeX
@InProceedings{Choudhuri_2023_CVPR,
    author    = {Choudhuri, Anwesa and Chowdhary, Girish and Schwing, Alexander G.},
    title     = {Context-Aware Relative Object Queries To Unify Video Instance and Panoptic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {6377-6386}
}
```



