GPUS=8
PY_FILE="../generate.py"
FLF2V_14B_14B_CKPT_DIR=Wan-AI/Wan2.1-FLF2V-14B-720P
# export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH


export ALGO=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun \
    --master_port=23152 \
    --nproc_per_node=$GPUS $PY_FILE \
    --task flf2v-14B \
    --ckpt_dir $FLF2V_14B_14B_CKPT_DIR \
    --size 960*960 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size $GPUS \
    --first_frame ../examples/flf2v_input_first_frame.png \
    --last_frame ../examples/flf2v_input_last_frame.png \
    --prompt "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。"