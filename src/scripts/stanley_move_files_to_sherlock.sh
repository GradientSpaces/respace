
# move dataset pickle files

scp ./data/cache/dataset_bedroom_train.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_bedroom_train.pkl
scp ./data/cache/dataset_bedroom_val.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_bedroom_val.pkl
scp ./data/cache/dataset_bedroom_test.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_bedroom_test.pkl

scp ./data/cache/dataset_livingroom_train.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_livingroom_train.pkl
scp ./data/cache/dataset_livingroom_val.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_livingroom_val.pkl
scp ./data/cache/dataset_livingroom_test.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_livingroom_test.pkl

scp ./data/cache/dataset_all_train.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_all_train.pkl
scp ./data/cache/dataset_all_val.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_all_val.pkl
scp ./data/cache/dataset_all_test.pkl sherlock:/scratch/users/mnbucher/stan24sgllm/data/dataset_all_test.pkl

# move full rendering folders

zip -r ./eval/viz/3d-front-train-full-scenes-all.zip ./eval/viz/3d-front-train-full-scenes-all
scp ./eval/viz/3d-front-train-full-scenes-all.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-full-scenes-all.zip

zip -r ./eval/viz/3d-front-train-full-scenes-bedroom.zip ./eval/viz/3d-front-train-full-scenes-bedroom
scp ./eval/viz/3d-front-train-full-scenes-bedroom.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-full-scenes-bedroom.zip

zip -r ./eval/viz/3d-front-train-full-scenes-livingroom.zip ./eval/viz/3d-front-train-full-scenes-livingroom
scp ./eval/viz/3d-front-train-full-scenes-livingroom.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-full-scenes-livingroom.zip

# move instr rendering folders

zip -r ./eval/viz/3d-front-train-instr-scenes-all.zip ./eval/viz/3d-front-train-instr-scenes-all
scp ./eval/viz/3d-front-train-instr-scenes-all.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-instr-scenes-all.zip

zip -r ./eval/viz/3d-front-train-instr-scenes-bedroom.zip ./eval/viz/3d-front-train-instr-scenes-bedroom
scp ./eval/viz/3d-front-train-instr-scenes-bedroom.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-instr-scenes-bedroom.zip

zip -r ./eval/viz/3d-front-train-instr-scenes-livingroom.zip ./eval/viz/3d-front-train-instr-scenes-livingroom
scp ./eval/viz/3d-front-train-instr-scenes-livingroom.zip sherlock:/scratch/users/mnbucher/stan24sgllm/viz/3d-front-train-instr-scenes-livingroom.zip

# move prompts

scp ./data/3D-FUTURE-assets/model_info_martin_prompts.json sherlock:/scratch/users/mnbucher/stan24sgllm/3dfront/3D-FUTURE-assets/model_info_martin_prompts.json
# scp ./data/3D-FUTURE-assets/model_info_martin_embeds.pickle sherlock:/scratch/users/mnbucher/stan24sgllm/3dfront/3D-FUTURE-assets/model_info_martin_embeds.pickle