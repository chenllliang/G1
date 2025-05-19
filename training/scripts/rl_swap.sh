
set -x
cd EasyR1

MODEL_PATH=PATH_TO_QWEN2_5_VL_7B_Instruct
EXPERIMENT_NAME=qwen2_5_vl_instruct_swap_rl

python3 -m verl.trainer.main_sandbox_match3 \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.temperature=1 \
    worker.rollout.n=5 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path="./ckpts/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=50 

