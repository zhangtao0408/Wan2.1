#!/bin/bash

# ================= 1. 环境配置 =================
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

# 设置当前目录到 python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 优化参数
export DECORD_EOF_RETRY_MAX=65536
export OMP_NUM_THREADS=12        
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'

# MindIE-SD 优化
export ALGO=1

# ================= 2. 分布式参数 =================
nnodes=1
nproc_per_node=8  # 使用 8 卡
node_rank=0
master_addr=localhost
port=29535

launch_args="--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port=$port"

# ================= 3. 模型与路径参数 =================
task="multitalk-14B"

# 路径变量名统一
ckpt_dir="multitalk_weights/Wan2.1-I2V-14B-480P"
w2v_dir="multitalk_weights/chinese-wav2vec2-base"
kokoro_dir="multitalk_weights/Kokoro-82M"


# 输入配置文件 (Prompt 和 Audio 路径都在这里面定义)
input_json="examples/multitalk_example_tts_1.json"

# 可选: multitalk-480 或 multitalk-720
size="multitalk-480" 

# ================= 4. 推理超参数 =================
num_frames=81
step=4            # 代码默认 40
cfg_text=1.0      # 文本引导系数 (Text Guidance Scale)
cfg_audio=4.0     # 音频引导系数 (Audio Guidance Scale)
shift=10.0        # 采样偏移
seed=42           # 固定种子方便复现， -1 用于随机
if [ "$seed" -eq "-1" ]; then
    seed=$RANDOM
fi

# 并行设置 
ulysses_size=8
ring_size=1

# ================= 5. profilling采集 ===========

export PROFILING_ENABLE=0       # 0: disable, 1: enable
export PROFILING_LEVEL=1        # 0: Level0, 1: Level1, 2: Level2
export PROFILING_DIR=./prof     # profiling dir (default: ./prof)
export PROFILING_PYTHON_STACK=0 # enable python stack (default: 0)

# >>>>>
# Currentyly, the profiling will warm up 1 step, wait 1 step, active 1 step, repeat 1 step, skip 0 step.
# <<<<<

# ================= 6. 启动命令 =================
echo "Starting MultiTalk Inference..."
echo "Task: $task | Size: $size | GPUs: $nproc_per_node"

torchrun $launch_args generate_multitalk.py \
    --task $task \
    --size $size \
    --frame_num $num_frames \
    --ckpt_dir $ckpt_dir \
    --wav2vec_dir $w2v_dir \
    --kokoro_dir $kokoro_dir \
    --input_json $input_json \
    --audio_mode tts \
    --sample_steps $step \
    --sample_shift $shift \
    --sample_text_guide_scale $cfg_text \
    --sample_audio_guide_scale $cfg_audio \
    --base_seed $seed \
    --ulysses_size $ulysses_size \
    --ring_size $ring_size \
    --dit_fsdp \
    --t5_fsdp \
    --offload_model False  # 8卡通常不需要 offload，显存不够改 True


# --input_json $input_json \
    # --dit_fsdp \
    # --t5_fsdp \
    #    --task $task \
    #    --vae_parallel \