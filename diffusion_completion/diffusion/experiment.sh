python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --epochs 100 \
  --lr 2e-4 \
  --T 100 \
  --experiment_name diffusion_1 \
  --val_sample_interval 10 

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --epochs 100 \
  --lr 2e-4 \
  --T 100 \
  --experiment_name diffusion_2 \
  --val_sample_interval 10 


