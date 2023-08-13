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


### Steps:

1. Please create a folder under your **./data** directory called **Cityscapes/**.

2. In **./data/Cityscapes/**, clone [this](https://github.com/mcahny/vps/tree/master) repository and follow the steps to install it's requirements.  

3. Download and prepare the Cityscapes-VPS dataset as described [here](https://github.com/mcahny/vps/blob/master/docs/DATASET.md).
After this step, dataset structure should look like the following.

```
data/
Cityscapes/
  cityscapes_vps/
    panoptic_im_train_city_vps.json
    panoptic_im_val_city_vps.json
    panoptic_im_test_city_vps.json  
    instances_train_city_vps_rle.json
    instances_val_city_vps_rle.json
    im_all_info_val_city_vps.json
    im_all_info_test_city_vps.json
    panoptic_gt_val_city_vps.json
    train/
      img/
      labelmap/
    val
      img/
      img_all/
      panoptic_video/
    test
      img_all/
```

4. Run the following under the caroq conda environment and CAROQ home directory:
```
./datasets/generate_cityscapes_vps_script.sh
```
After this step, 3 new json files are generated as follows.
```
data/
  Cityscapes/
    cityscapes_vps/
      panoptic_vps_train.json
      panoptic_vps_val.json
      panoptic_vps_test.json
```
Now we are got to run training/evaluation.


## Expected dataset structure for [KITTI-MOTS](https://www.cvlibs.net/datasets/kitti/eval_mots.php), [MOTS-2020](https://motchallenge.net/workshops/bmtt2020/tracking.html)

```
data/
  {KITTI_MOTS, MOTS_2020}/
    train/
      images/
      instances_txt/
    val/
      images/
      instances_txt/
    train_full.json
    val_full.json
```

### Steps to create KITTI-MOTS:

1. Please create a folder under your **./data** directory called **KITTI_MOTS/**. Download the KITTI-MOTS images and annotations from [here](https://www.vision.rwth-aachen.de/page/mots) under this directory.
   
2. Run the following to create a usable format of the dataset.
   ```cd datasets/
      python generate_kittimots.py
   ```

### Steps to create MOTS-2020:

1. Please create a folder under your **./data** directory called **MOTS_2020/**. Download the MOTS-2020 images and annotations from [here](https://www.vision.rwth-aachen.de/page/mots) under this directory.
   
2. Run the following to create a usable format of the dataset.
   ```cd datasets/
      python generate_mots2020.py
   ```
