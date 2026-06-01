# sglang 昇腾 NPU 安装指南

> 官方文档参考：https://docs.sglang.com.cn/platforms/ascend_npu.html

---

## 前置条件

在开始安装前，请确保已满足以下条件：

| 依赖项 | 要求 | 说明 |
|--------|------|------|
| CANN 版本 | ≥ 8.2.0.0 | 昇腾工具包 |
| Python 版本 | 3.11 | 推荐版本 |
| 设备类型 | A3/Ascend 910B | 当前支持的 NPU 设备 |

---

## 安装步骤

### 1. 安装 sglang

```bash
(
    # 克隆源码并安装
    git clone https://github.com/Yanguan619/sglang.git -b minicpm_sala_dense --depth 1
    cd sglang
    mv python/pyproject_other.toml python/pyproject.toml
    pip install -e python[srt_npu]
)
```

### 2. 安装 triton-ascend

```bash
pip install triton-ascend==3.2.0.dev20260507 --no-cache-dir -i https://test.pypi.org/simple/
```

### 3. 安装 sgl-kernel-npu

```bash
(
    # 克隆并构建
    git clone https://github.com/sgl-project/sgl-kernel-npu.git
    cd sgl-kernel-npu
    git checkout 2026.05.01
    git submodule update --init --recursive
    bash build.sh

    # 安装生成的 wheel 包
    pip install output/sgl_kernel_npu*.whl --force
    pip install output/deep_ep-*.whl --force

    # 运行测试
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 tests/python/sgl_kernel_npu/test_hello_world.py
)
```

### 4. 安装 flash-linear-attention

```bash
(
    cd /home/minicpmsala
    # 下载并安装 fla
    wget https://gh.llkk.cc/https://github.com/fla-org/flash-linear-attention/archive/refs/tags/v0.4.2.tar.gz --no-check-certificate
    tar -xvf v0.4.2.tar.gz
    cd flash-linear-attention-0.4.2
    pip install -e . --no-deps
    cd ..
)
```

### 5. 安装 unum_ops (minicpm_sala_dense CustomOps)

```bash
(
    git clone https://github.com/Yanguan619/unum_ops.git --depth 1
    cd unum_ops
    pip install -e . --no-deps
)
```

### 6. 安装 CustomOps (可选)

```bash
# 设置设备类型（根据实际设备修改）
export DEVICE_TYPE="a3"

# 下载并安装 CANN custom ops
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-${DEVICE_TYPE}-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.2.0.0-${DEVICE_TYPE}-linux.aarch64.run
./CANN-custom_ops-8.2.0.0-${DEVICE_TYPE}-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp

# 安装 Python custom ops
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.${DEVICE_TYPE}-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-1.0.${DEVICE_TYPE}-cp311-cp311-linux_aarch64.whl
```

---

## 环境检查

安装完成后，运行以下命令验证环境配置是否正确：

```bash
python3 -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
python3 -c "from sglang.srt.models.minicpm import MiniCPMSALAForCausalLM"
```

---

## 服务启动

### 权重下载

```bash
modelscope download --model OpenBMB/MiniCPM-SALA --local_dir /data2/OpenBMB/MiniCPM-SALA
```

### 环境变量配置

```bash
# 网络配置
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Pytorch 内存配置
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# sglang 配置
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0  # error -> warn

export ASCEND_RT_VISIBLE_DEVICES=6
python3 -m sglang.launch_server \
    --model-path /data2/OpenBMB/MiniCPM-SALA/ \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend ascend \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --context-length 132000 \
    --max-total-tokens 132000 \
    --mem-fraction-static 0.9

### 启动命令
python3 -m sglang.launch_server \
    --model-path /data2/OpenBMB/MiniCPM-SALA/ \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend ascend \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --skip-server-warmup \
    --context-length 132000 \
    --max-total-tokens 132000 \
    --mem-fraction-static 0.9 \
    --disable-cuda-graph
```

---

## 性能分析（Profiling）

### 步骤说明

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 启动 profiling | 开始采集 CPU 和 NPU 性能数据 |
| 2 | 发送请求 | 产生推理负载 |
| 3 | 停止 profiling | 停止采集并生成报告 |

### 执行命令

```bash
# Step 1: 启动性能分析
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "./sglang_profile",
    "start_step": 1,
    "activities": ["CPU", "NPU"]
  }'

# Step 2: 发送测试请求
curl http://127.0.0.1:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}'

curl http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 64,
        "ignore_eos": true
    }'

# Step 3: 停止性能分析
curl -X POST http://127.0.0.1:30000/stop_profile
```

---

## API 测试

### 测试命令

```bash
# 测试默认模型（MiniCPM-SALA）
curl http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "你好,请介绍下你自己"}],
        "max_tokens": 64,
        "temperatures": 0.0,
        "ignore_eos": true
    }'

curl -X POST "http://127.0.0.1:30000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
    "model": "default",
    "messages": [
        {
        "role": "system",
        "content": "你的任务是回答用户的问题，不需要有思考过程，每次回答时先介绍下自己。"
        },
        {
        "role": "user",
        "content": "你是一个什么类型的模型"
        }
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "ignore_eos": true
    }'

evalscope perf \
    --model default \
    --url http://127.0.0.1:30000/v1/completions --port 30000 --api-key "API_KEY" \
    --api openai \
    --parallel 1 --number 4 \
    --min-tokens 100 --max-tokens 100 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --dataset random \
    --prefix-length 0 \
    --tokenizer-path /data2/OpenBMB/MiniCPM-SALA \
    --extra-args '{"ignore_eos": true}'
```

---
