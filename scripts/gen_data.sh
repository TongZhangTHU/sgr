save_path=data/train # or data/test
tasks=open_microwave
episodes_per_task=100

renderer=opengl 

# In SGRv2, we only use the first variation for tasks with multiple variations, 
# unlike Peract, which mixes all variations.
variations=1 
all_variations=False

python libs/RLBench/tools/dataset_generator.py --save_path=$save_path --tasks=$tasks \
    --image_size=128,128  --renderer=$renderer --episodes_per_task=$episodes_per_task \
    --variations=$variations --all_variations=$all_variations