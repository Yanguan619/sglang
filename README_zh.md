<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

> [!WARNING]
> **MiniCPM-Sala 定制版本**
>
> 本仓库是 SGLang 针对 **OpenBMB MiniCPM-Sala** 模型优化的定制版本。
> 请 **仅** 使用此版本来运行 MiniCPM-Sala 模型。
>
> **环境依赖:**
> 本项目需要配合以下仓库共同构成完整的运行环境：
> 1. **InfLLM V2 CUDA Kernels**: [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl/tree/minicpm_sala) (已包含为 submodule，分支: `minicpm_sala`)
> 2. **Sparse Kernels**: [OpenBMB/sparse_kernel](https://github.com/OpenBMB/sparse_kernel) (已包含为 submodule)

--------------------------------------------------------------------------------

[English](./README.md) | [**中文**]

# MiniCPM-SALA 推理环境搭建流程

## 环境要求

- CUDA 12.x 或更高版本
- `gcc` / `g++` 编译器
- `uv` 包管理器（脚本会自动检测）

## 快速开始

### 安装

```bash
# 克隆仓库
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# 一键安装（自动创建虚拟环境并编译所有依赖）
bash install_minicpm_sala.sh

# 或指定 PyPI 镜像源
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

安装脚本会自动完成以下步骤：

1. 创建 `sglang_minicpm_sala_env` 虚拟环境（Python 3.12）
2. 克隆依赖到 `3rdparty/` 目录 (infllmv2) 并初始化子模块 (sparse_kernel)
3. 安装 MiniCPM-SALA (当前仓库)
4. 编译安装 `infllmv2_cuda_impl`
5. 编译安装 `sparse_kernel`
6. 安装 `tilelang` 和 `flash-linear-attention`

### 使用

```bash
# 激活环境
source sglang_minicpm_sala_env/bin/activate
export ASCEND_RT_VISIBLE_DEVICES="4" # (可选) 指定使用的 GPU 设备
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 # error->warn
# 启动推理服务（将 MODEL_PATH 替换为实际模型路径）
MODEL_PATH=/data/weights/MiniCPM-SALA/

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 2 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| 参数 | 说明 |
|------|------|
| `--trust-remote-code` | 允许加载模型自带的自定义代码 |
| `--disable-radix-cache` | 禁用 RadixAttention 前缀缓存 |
| `--attention-backend minicpm_flashinfer` | 使用 MiniCPM FlashInfer 注意力后端 |
| `--chunked-prefill-size 8192` | chunked prefill 大小 |
| `--max-running-requests 32` | 最大并发推理请求数 |
| `--skip-server-warmup` | 跳过服务预热 |
| `--port 31111` | 服务端口 |
| `--dense-as-sparse` | 使用 dense-as-sparse 模式 |

> **提示：** 为获得最佳生成效果，建议在请求时设置 `temperature=0.9`。

### 工具调用（Tool Calling）

启动服务时添加 `--tool-call-parser minicpm4_xml` 参数即可启用工具调用：

```bash
# 激活环境
source sglang_minicpm_sala_env/bin/activate

# 启动推理服务（启用工具调用，将 MODEL_PATH 替换为实际模型路径）
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse \
    --tool-call-parser minicpm4_xml
```

**请求示例：**

```bash
curl -X POST "http://localhost:31111/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm4.6-8b",
    "messages": [{"role": "user", "content": "北京天气怎么样"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }]
  }'
```

## 目录结构

```
sglang/
├── README.md                       # 英文文档
├── README_zh.md                    # 本文件（中文文档）
├── install_minicpm_sala.sh         # 一键安装脚本
├── 3rdparty/                       # 依赖源码
│   ├── infllmv2_cuda_impl/         # InfLLM v2 CUDA 实现
│   └── sparse_kernel/              # sparse_kernel_extension
├── python/                         # SGLang 源码
└── ...
```

## 手动安装

如果一键脚本不满足需求，可以分步执行：

```bash
# 0. 确保 uv 可用
pip install uv

# 1. 创建虚拟环境
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. 安装 SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. 编译安装 CUDA 扩展
# (确保依赖已克隆到 3rdparty/ 目录)
cd 3rdparty/infllmv2_cuda_impl && python setup.py install && cd ../..
cd 3rdparty/sparse_kernel && python setup.py install && cd ../..

# 4. 安装额外依赖
uv pip install tilelang flash-linear-attention
```

## Q&A

**Q: CUDA 扩展编译失败怎么办？**

- 确保系统安装了 CUDA 12 以上（`nvcc --version` 检查）。
- 确保 `gcc` / `g++` 可用。
- 如果 `CXX` 环境变量被设为 `clang++ -pthread`，手动 `export CXX=g++`。
