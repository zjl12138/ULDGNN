# ULDGNN
## data preparation
step 1 extract information from .sketch files <br>
`python read_info_from_sketch/read_sketch_multithread.py --dataDir=<your_path_to_sketch_files> --outDir=<you_dir>` <br>
step 2 generate dataset. You need to edit generation_fine.py to set correct rootdir and outDir<br>
`python lib/generation_fine.py`<br>
step 3 refine dataset to support RoiAlign module, which is used to refine predicted boxes. Remember to edit gen_ulgnn_data.py to set correct data_dir and outdir<br> 
`python gen_ulgnn_data.py --exp_name=gen_uldgnn` 

## Training
`nohup python -m torch.distributed.launch --nproc_per_node=4  train.py --exp_name=<Your_exp_name> --train_mode=1 --cfg_file=configs/<your_config_file>.yaml --epochs=50 --lr=0.0001 > <your_log>.out` <br>
`nohup python -m torch.distributed.launch --nproc_per_node=4  train.py --exp_name=<Your_exp_name> --train_mode=0 --cfg_file=configs/<your_config_file>.yaml --epochs=50 --lr=0.0001 > <your_log>.out` <br>
[optional] <br>
`nohup python -m torch.distributed.launch --nproc_per_node=4  train.py --exp_name=<Your_exp_name> --train_mode=2 --cfg_file=configs/<your_config_file>.yaml --epochs=50 --lr=0.0001 > <your_log>.out`
## Test
Edit test.py to optionally visualize results and get evaluation results <br>
`python test.py --exp_name=<training_exp_name> --cfg_file=configs/<training_config_file>.yaml` <br>