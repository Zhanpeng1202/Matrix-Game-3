torchrun --nproc_per_node=8 generate.py\
         --size "704*1280"\
         --dit_fsdp --t5_fsdp\
         --ckpt_dir Matrix-Game-3.0 --fa_version 3\
         --use_int8 --num_iterations 12 --num_inference_steps 3\
         --image demo_images/001/image.png\
         --prompt "A colorful, animated cityscape with a gas station and various buildings."\
         --save_name test_1 --seed 42 --compile_vae\
         --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output