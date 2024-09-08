tasks=open_microwave
demos=5
demo_path=data # path to the dataset
batch_size=16

start_seed=0
logdir=logs # path to save logs
training_iterations=20000 # increase training_iterations when number of demos increase
save_freq=$((training_iterations/25))

method=SGR
lr=0.003
color_drop=0.4
use_semantic=[5]
pretrained_model=clip
tag=sgrv2-demos_${demos}-iter_${training_iterations}

model=pointnext-xl_seg
mlps=[256]
num_classes=256

test_type=missing5 # eval last 5 checkpoints
test_envs=5
test_episodes=50

i=0 # GPU id

# Training
CUDA_VISIBLE_DEVICES=$i python train.py \
    rlbench.tasks=$tasks rlbench.demos=$demos rlbench.demo_path=$demo_path/train replay.batch_size=$batch_size \
    framework.start_seed=$start_seed framework.logdir=$logdir framework.save_freq=$save_freq framework.training_iterations=$training_iterations \
    method=$method method.lr=$lr method.color_drop=$color_drop method.use_semantic=$use_semantic method.pretrained_model=$pretrained_model \
    method.tag=$tag model=$model model.cls_args.mlps=$mlps model.cls_args.num_classes=$num_classes

# Evaluation
CUDA_VISIBLE_DEVICES=$i python eval.py \
    rlbench.tasks=$tasks rlbench.demo_path=$demo_path/test framework.start_seed=$start_seed framework.logdir=$logdir \
    framework.eval_type=$test_type framework.eval_envs=$test_envs framework.eval_episodes=$test_episodes \
    method.name=$method method.tag=$tag model.name=$model
