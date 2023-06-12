# Example script for training and evaluation

## For training

CUDA_VISIBLE_DEVICES=0,1,2,3, python train_net.py \
              --config-file configs/ovis/R50.yaml \
              --num-gpus 4 \
              OUTPUT_DIR saved_outputs/ovis


## For evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3, python train_net.py \
              --config-file configs/ovis/R50.yaml \
              --num-gpus 4 --eval-only \
              OUTPUT_DIR saved_outputs/ovis
