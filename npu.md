```bash
IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:v0.18.0rc1-a3
MODEL=/data/
docker run -itd --name "$(date +%Y-%m-%d)" \
	--privileged \
	--ipc=host \
	--net=host \
	--shm-size=500g \
	-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
	-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
	-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
	-v /usr/local/dcmi:/usr/local/dcmi \
	-v /usr/local/sbin:/usr/local/sbin \
	-v /etc/hccn.conf:/etc/hccn.conf \
	-v $MODEL:$MODEL \
	-e VLLM_USE_MODELSCOPE=true \
	-e ASCEND_RT_VISIBLE_DEVICES="0,1" \
	-e HCCL_OPEXPANSIONMODE="AIV" \
	-e HCCL_BUFFSIZE="1024" \
	-e OMP_PROC_BIND="false" \
	-e OMP_NUM_THREADS="1" \
	-e TASK_QUEUE_ENABLE="1" \
    $IMAGE bash
```

# https://docs.sglang.com.cn/platforms/ascend_npu.html
# fla
```bash
(
    wget https://github.com/fla-org/flash-linear-attention/archive/refs/tags/v0.4.2.tar.gz --no-check
)
```
# sgl-kernel-npu
```bash
(
    git clone https://github.com/sgl-project/sgl-kernel-npu && \
    cd sgl-kernel-npu && \
    bash build.sh
    pip install output/sgl_kernel_npu*.whl
     pip install output/deep_ep-*.whl
    # (Optional) Confirm whether the import can be successfully
    python3 -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 tests/python/sgl_kernel_npu/test_hello_world.py
)
```
# CustomOps
```bash
(
    DEVICE_TYPE="a3"
    wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
    chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
    ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
    wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
    pip install ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
)
```
# sglang
```bash
(
    cd sglang-minicpm
    # mv python/pyproject_other.toml python/pyproject.toml
    pip install -e python[srt_npu]
)
```
# serve
```bash
#!/bin/bash
set -euo pipefail
python3 -c "import sglang.srt.models.minicpm; print('OK -', sglang.srt.models.minicpm.MiniCPMSALAForCausalLM)"

MODEL_PATH=/data/weights/MiniCPM-SALA/
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# sglang
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 # error->warn
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --disable-radix-cache \
    --disable-cuda-graph \
    --attention-backend ascend \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --skip-server-warmup \
    --context-length 132000 \
    --max-total-tokens 132000 \
    --mem-fraction-static 0.9
```

# curl

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{ "model": "default", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 64 }'


curl http://127.0.0.1:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{ "model": "qwen3.5", "temperature": 0.0, "messages": [{"role": "user", "content": "你好"}], "max_tokens": 4 }'
```
