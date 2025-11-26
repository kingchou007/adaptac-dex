cd Dex-Sense
export HYDRA_FULL_ERROR=1 

python eval.py \
    --ckpt /home/jinzhou/Lab/RISE-dex/ckpt/openbox/open_box_010401/policy_epoch_10000_seed_233.ckpt \
    --calib /home/jinzhou/Lab/RISE-dex/Dex-Sense/data/constants \
    --num_action 20 \
    --num_inference_step 4 \
    --voxel_size 0.005 \
    --obs_feature_dim 512 \
    --hidden_dim 512 \
    --nheads 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 1 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --max_steps 600 \
    --seed 233 \
    --discretize_rotation \
    --ensemble_mode act \
    --num_obs 1 \
    --use_color \
    --repr_frame camera \
    --vis \
   
