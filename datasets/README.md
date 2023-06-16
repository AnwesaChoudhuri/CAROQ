# Prepare Datasets
We follow Detectron2 to register custom datasets, use a custom dataset mapper and a custom dataset loader.
Please create a new directory called **./data** in the current working directory and download all the datasets. Under this directory, detectron2 will look for datasets in the structure described below:

```
data/
  Youtube_vis_2019/
  Youtube_vis_2021/
  OVIS/
  Cityscapes/
  KITTI-MOTS/
  MOTS-2020/
```

## Expected dataset structure for VIS datasets ([Youtube-VIS 2019](https://competitions.codalab.org/competitions/20128), [Youtube-VIS 2021](https://competitions.codalab.org/competitions/28988), [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate)).

Same as [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets). 
The structure of each dataset should be as follows.

```
data/
{dataset_name}/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [Cityscapes-VPS](https://opendatalab.com/Cityscapes-VPS/github.com/mcahny/vps)
Coming soon...

## Expected dataset structure for [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/eval_mots.php), [MOTS-2020](https://motchallenge.net/workshops/bmtt2020/tracking.html)
Coming soon...
