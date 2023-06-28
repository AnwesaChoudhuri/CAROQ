for mode in "train" "val" "test"
do
  python datasets/generate_cityscapes_vps.py $mode
done
