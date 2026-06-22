# llama.cpp CUDA 视觉模型性能排查与调优报告

## 1. 文档信息

- 测试日期：2026-06-22
- 时区：Asia/Shanghai（UTC+8）
- 工作目录：`G:\llama-b9672-bin-win-cuda-13.3-x64`
- 目标：
  1. 查明实际生成速度只有约 12 token/s 的原因。
  2. 补齐 CUDA 13.3 运行依赖。
  3. 保证主模型和视觉投影模型均使用 GPU。
  4. 在 RTX 3060 12GB 上寻找当前模型的最快可用参数。
  5. 优化真实图片请求的首轮等待时间。

本报告记录本次排查中使用过的参数、观测数据、测试结果、最终配置、已知风险及回滚方法。

---

## 2. 测试环境

### 2.1 硬件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 |
| 显存 | 12288 MiB（程序报告为 12287 MiB） |
| NVIDIA 驱动 | 610.62 |
| CPU | Intel Core i5-12400F |
| CPU 核心/线程 | 6 核 / 12 线程 |
| 系统内存 | 32 GiB（34359738368 bytes） |
| 内存频率 | DDR4-2666 |

### 2.2 llama.cpp

```text
version: 9672 (74ade5274)
built with Clang 20.1.8 for Windows x86_64
```

目录是预编译的 Windows CUDA 运行包，不是源码仓库，也不是 Git 工作树。

### 2.3 模型文件

| 文件 | 字节数 | 约合 GiB | 用途 |
|---|---:|---:|---|
| `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf` | 11659235456 | 10.86 | 主语言模型，35B MoE / A3B，IQ2_M |
| `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf` | 15440519296 | 14.38 | 更高精度主模型，本次最终未使用 |
| `mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf` | 899283072 | 0.84 | 视觉投影模型 |

模型目录总大小约 26.08 GiB。

### 2.4 CUDA 运行库

使用的依赖压缩包：

```text
C:\Users\27836\Downloads\cudart-llama-bin-win-cuda-13.3-x64.zip
```

SHA-256：

```text
1462A050EB4C684921BA51DCC4CC488A036674C3E73E9945EE705B854808D03E
```

压缩包内容：

| 文件 | 字节数 |
|---|---:|
| `cublas64_13.dll` | 51870320 |
| `cublasLt64_13.dll` | 460301424 |
| `cudart64_13.dll` | 551024 |

---

## 3. 原始问题与最初配置

最初观察到的生成速度约为 12 token/s。最初脚本曾包含：

```bat
llama-server.exe ^
-m "models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf" ^
--mmproj "models\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf" ^
-ngl 999 ^
--flash-attn on ^
--cache-type-k q8_0 ^
--cache-type-v q8_0 ^
-c 64000 ^
-n 4096 ^
--host 127.0.0.1 ^
--port 8080
```

后续测试前的脚本已缩减为：

```bat
llama-server.exe ^
-m "models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf" ^
--mmproj "models\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf" ^
-ngl 999 ^
--flash-attn on ^
--cache-type-k q8_0 ^
--cache-type-v q8_0 ^
-c 8192 ^
-n 4096 ^
-np 1 ^
--host 127.0.0.1 ^
--port 8080
```

该版本已备份为：

```text
G:\llama-b9672-bin-win-cuda-13.3-x64\start.bat.before-tuning.bak
```

---

## 4. 第一阶段：确认 12 token/s 的根因

### 4.1 CUDA 后端未加载时的现象

最初运行：

```powershell
.\llama-server.exe --list-devices
```

输出只有：

```text
Available devices:
```

程序启动日志只注册 CPU：

```text
device_info:
  - CPU : 12th Gen Intel(R) Core(TM) i5-12400F
```

对 `ggml-cuda.dll` 使用 Windows `LoadLibrary` 测试，返回：

```text
ggml-cuda.dll failed, Win32 error 126
```

错误 126 表示 DLL 本身或其依赖项无法找到。

从 `ggml-cuda.dll` 的导入字符串中识别出的关键依赖包括：

```text
cublas64_13.dll
nvcudart_hybrid64.dll
nvcuda.dll
MSVCP140.dll
VCRUNTIME140.dll
VCRUNTIME140_1.dll
```

其中系统中未找到 `cublas64_13.dll`。这是 CUDA 后端未加载的主要原因。

### 4.2 CPU 基准

测试命令：

```powershell
.\llama-bench.exe `
  -m .\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf `
  -ngl 999 `
  -p 512 `
  -n 128 `
  -r 1 `
  -ctk q8_0 `
  -ctv q8_0 `
  -fa on `
  -o md
```

结果：

| 后端 | 测试 | 速度 |
|---|---|---:|
| CPU | `pp512` | 26.12 token/s |
| CPU | `tg128` | 11.65 token/s |

关键输出：

```text
backend: CPU
tg128: 11.65 token/s
```

该结果与用户实际观察到的约 12 token/s 几乎完全一致，因此确认当时虽然写了 `-ngl 999`，但实际仍是纯 CPU 推理。

---

## 5. 第二阶段：安装 CUDA 13.3 依赖

### 5.1 安装操作

对下载的压缩包执行了以下步骤：

1. 检查文件大小和 SHA-256。
2. 枚举 ZIP 内文件，确认包含三个 CUDA 13 运行库。
3. 将 ZIP 解压到临时目录。
4. 将以下文件复制到 `llama-server.exe` 同目录：

```text
G:\llama-b9672-bin-win-cuda-13.3-x64\cublas64_13.dll
G:\llama-b9672-bin-win-cuda-13.3-x64\cublasLt64_13.dll
G:\llama-b9672-bin-win-cuda-13.3-x64\cudart64_13.dll
```

目标目录原先没有这三个同名文件，因此本次没有产生 DLL 覆盖备份。

### 5.2 安装后验证

`ggml-cuda.dll` 加载测试：

```text
ggml-cuda.dll loaded successfully
```

设备枚举：

```text
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11250 MiB free)
```

详细日志：

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 12287 MiB)
Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
```

由此确认 CUDA 后端已真正启用。

---

## 6. CUDA 启用后的初始基准

测试命令：

```powershell
.\llama-bench.exe `
  -m .\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf `
  -ngl 999 `
  -p 128 `
  -n 64 `
  -r 1 `
  -ctk q8_0 `
  -ctv q8_0 `
  -fa on `
  -o md
```

结果：

| 后端 | 测试 | 速度 |
|---|---|---:|
| CUDA | `pp128` | 478.04 token/s |
| CUDA | `tg64` | 19.01 token/s |

这一步证明 CUDA 已生效，但它是 `llama-bench` 的短生成测试，不包含服务端、视觉投影、真实 API 调度及长输出，不能直接代表最终服务速度。

---

## 7. 服务端曾出现约 9 token/s 的原因

曾使用的服务配置：

```text
主模型：IQ2_M，约 10.85 GiB
mmproj：F16，约 0.84 GiB
上下文：16384
KV Cache：Q8_0 / Q8_0
并行槽位：自动，实际为 4
```

实时状态：

```text
GPU 显存使用：11950 MiB
GPU 显存剩余：161 MiB
total_slots：4
每个 slot 的 n_ctx：16384
```

服务日志中观察到：

```text
n_decoded = 100, tg = 9.46 t/s
n_decoded = 129, tg = 9.47 t/s
n_decoded = 157, tg = 9.39 t/s
n_decoded = 181, tg = 9.14 t/s
n_decoded = 204, tg = 8.91 t/s
```

判断：

- 模型权重、F16 mmproj、KV Cache、CUDA 工作缓冲和桌面程序共同挤压 12GB 显存。
- 默认自动创建 4 个 slot，增加服务端上下文/KV 管理压力。
- 显存只剩约 161 MiB，VMM/共享显存/PCIe 内存访问可能造成明显抖动。
- 因此出现“CUDA 已启用但反而比 CPU 还慢”的现象。

对应修正方向：

```text
-np 1
-c 8192 或 4096
保留 Q8 KV
保留 mmproj GPU offload
```

---

## 8. KV Cache 类型测试结论

曾建议过使用：

```text
--cache-type-k q4_0
--cache-type-v q4_0
```

目的是节省显存，但用户实际测试发现 Q4_0 比 Q8_0 更慢。

原因判断：

- Q4 KV 虽减少容量，但增加 KV 解量化和访问开销。
- RTX 3060 对当前 llama.cpp 构建、模型结构和 Q8 路径的综合表现更好。
- 当前瓶颈并非只由 KV 容量决定。

因此最终保留：

```text
--cache-type-k q8_0
--cache-type-v q8_0
```

本报告没有取得用户 Q4 测试的完整数值日志，因此只记录定性结论：**Q4_0 实测慢于 Q8_0**。

---

## 9. `common_fit_params` 警告说明

出现过：

```text
W common_fit_params: failed to fit params to free device memory:
n_gpu_layers already set by user to 999, abort
```

含义：

- `--fit` 默认开启。
- 程序尝试根据可用显存自动调整参数。
- 用户又明确设置了 `-ngl 999`。
- 自动适配器不能覆盖用户显式指定的 GPU 层数，因此中止适配并输出警告。

它不表示 CUDA 失效，但意味着自动显存适配没有完成。

后续测试分别比较了：

```text
-ngl all --fit off
-ngl auto -fitt 256
-ngl auto -fitt 512
```

---

## 10. 正式调参方法

### 10.1 固定条件

所有正式服务端参数测试均固定：

```text
模型：IQ2_M
视觉投影：F16 mmproj
mmproj GPU offload：开启
Flash Attention：开启
KV Cache K：Q8_0
KV Cache V：Q8_0
并行 slot：1
HTTP 地址：127.0.0.1
测试端口：8081
```

公共启动参数：

```text
-m <IQ2_M model>
--mmproj <F16 mmproj>
--mmproj-offload
--flash-attn on
-np 1
-n 256
--host 127.0.0.1
--port 8081
```

### 10.2 文本 API 测试请求

接口：

```text
POST /v1/chat/completions
```

主要测试提示：

```text
Write a numbered list of practical methods for improving software reliability.
Keep every item concise and continue until the token limit.
```

长测参数：

```json
{
  "temperature": 0.7,
  "max_tokens": 384,
  "ignore_eos": true,
  "stream": false
}
```

正式长测前先发送 16-token 短请求用于 CUDA kernel 和分配器预热。

指标说明：

- `prompt eval time`：提示处理速度。
- `eval time`：生成阶段总时间。
- `tg`：运行期间日志给出的 token generation 速度。
- `WallTokensPerSecond`：API completion token 数除以端到端请求时间。
- `VramMiB`：测试期间 `nvidia-smi` 报告的整卡显存使用量，不等于单进程独占显存。

---

## 11. 第一轮参数矩阵

### 11.1 参数组合

| 编号 | 名称 | GPU 层/适配 | 上下文 | KV | slot |
|---:|---|---|---:|---|---:|
| 1 | `all-q8-ctx8192` | `-ngl all --fit off` | 8192 | Q8/Q8 | 1 |
| 2 | `auto256-q8-ctx8192` | `-ngl auto -fitt 256` | 8192 | Q8/Q8 | 1 |
| 3 | `all-q8-ctx4096` | `-ngl all --fit off` | 4096 | Q8/Q8 | 1 |
| 4 | `auto256-q8-ctx4096` | `-ngl auto -fitt 256` | 4096 | Q8/Q8 | 1 |
| 5 | `auto512-q8-ctx4096` | `-ngl auto -fitt 512` | 4096 | Q8/Q8 | 1 |

### 11.2 第一轮短测结果

| 参数组合 | 提示 token | 生成 token | 提示速度 | 生成速度 | 总时间 |
|---|---:|---:|---:|---:|---:|
| `all-q8-ctx8192` | 33 | 192 | 223.66 t/s | 66.78 t/s | 3022.71 ms |
| `auto256-q8-ctx8192` | 33 | 192 | 135.62 t/s | 53.77 t/s | 3813.76 ms |
| `all-q8-ctx4096` | 33 | 192 | 95.81 t/s | 16.37 t/s | 12072.94 ms |
| `auto256-q8-ctx4096` | 33 | 192 | 137.02 t/s | 54.05 t/s | 3792.97 ms |
| `auto512-q8-ctx4096` | 33 | 192 | 127.89 t/s | 52.57 t/s | 3910.41 ms |

短测期间日志中的阶段值：

| 参数组合 | 中间 `tg` |
|---|---:|
| `all-q8-ctx8192` | 67.34 t/s（100 decoded） |
| `auto256-q8-ctx8192` | 53.84 t/s（100 decoded） |
| `all-q8-ctx4096` | 16.33 → 16.26 t/s |
| `auto256-q8-ctx4096` | 53.73 t/s（100 decoded） |
| `auto512-q8-ctx4096` | 52.73 t/s（100 decoded） |

### 11.3 第一轮观察

1. 自动适配组合稳定在约 52.6–54.1 t/s。
2. `-fitt 256` 比 `-fitt 512` 略快。
3. 自动适配下，4096 和 8192 上下文的文本生成速度几乎相同。
4. `all-q8-ctx4096` 的一次运行异常低，只有 16.37 t/s。
5. `all-q8-ctx8192` 却达到约 66.8 t/s。
6. 这说明强制全层并关闭适配时，对当时显存布局、GPU 时钟、VMM 状态或后台 GPU 占用非常敏感，不能只依据一次短测下结论。

---

## 12. 第二轮长输出复测

为了减少短输出、EOS 和 GPU 时钟爬升带来的误差，将候选组合改为：

```text
max_tokens = 384
ignore_eos = true
```

### 12.1 长测结果汇总

| 参数组合 | 状态 | 日志生成速度 | API 墙钟速度 | 生成 token | 整卡显存使用 |
|---|---|---:|---:|---:|---:|
| `all-q8-ctx8192` | 成功 | 66.25 t/s | 64.04 t/s | 384 | 11873 MiB |
| `auto256-q8-ctx8192` | 成功 | 53.59 t/s | 51.41 t/s | 384 | 11631 MiB |
| `auto256-q8-ctx4096` | 成功 | 53.59 t/s | 51.47 t/s | 384 | 11585 MiB |

### 12.2 `all-q8-ctx8192` 详细数据

参数：

```text
-ngl all
--fit off
-c 8192
-ctk q8_0
-ctv q8_0
-np 1
```

预热：

```text
prompt eval: 16 tokens / 178.39 ms = 89.69 t/s
generation: 16 tokens / 250.12 ms = 63.97 t/s
total: 428.51 ms / 32 tokens
```

正式长测：

```text
n_decoded = 100, tg = 67.00 t/s
n_decoded = 298, tg = 66.25 t/s
prompt eval: 33 tokens / 142.29 ms = 231.93 t/s
generation: 384 tokens / 5799.04 ms = 66.22 t/s
total: 5941.33 ms / 417 tokens
API wall throughput: 64.04 completion token/s
VRAM used: 11873 MiB
```

### 12.3 `auto256-q8-ctx8192` 详细数据

参数：

```text
-ngl auto
-fitt 256
-c 8192
-ctk q8_0
-ctv q8_0
-np 1
```

程序报告：

```text
estimated worst-case memory usage of mmproj: 1130.63 MiB
fitting params to device memory ...
```

预热：

```text
prompt eval: 16 tokens / 249.22 ms = 64.20 t/s
generation: 16 tokens / 322.27 ms = 49.65 t/s
total: 571.49 ms
```

正式长测：

```text
n_decoded = 100, tg = 53.84 t/s
n_decoded = 261, tg = 53.59 t/s
prompt eval: 33 tokens / 242.35 ms = 136.17 t/s
generation: 384 tokens / 7171.48 ms = 53.55 t/s
total: 7413.82 ms / 417 tokens
API wall throughput: 51.41 completion token/s
VRAM used: 11631 MiB
```

### 12.4 `auto256-q8-ctx4096` 详细数据

参数：

```text
-ngl auto
-fitt 256
-c 4096
-ctk q8_0
-ctv q8_0
-np 1
```

程序报告：

```text
estimated worst-case memory usage of mmproj: 1130.63 MiB
fitting params to device memory ...
```

预热：

```text
prompt eval: 16 tokens / 249.69 ms = 64.08 t/s
generation: 16 tokens / 313.91 ms = 50.97 t/s
total: 563.61 ms
```

正式长测：

```text
n_decoded = 100, tg = 53.76 t/s
n_decoded = 261, tg = 53.59 t/s
prompt eval: 33 tokens / 243.77 ms = 135.38 t/s
generation: 384 tokens / 7164.57 ms = 53.60 t/s
total: 7408.34 ms / 417 tokens
API wall throughput: 51.47 completion token/s
VRAM used: 11585 MiB
```

### 12.5 长测结论

最快组合：

```text
-ngl all
--fit off
-c 8192
Q8 KV
-np 1
```

相对 `auto256-q8-ctx8192`：

```text
日志生成速度提升：
(66.25 - 53.59) / 53.59 ≈ 23.62%

API 墙钟速度提升：
(64.04 - 51.41) / 51.41 ≈ 24.57%
```

代价：

```text
显存从 11631 MiB 增加到 11873 MiB
增加约 242 MiB
整卡只剩约 415 MiB 余量
```

因此该配置追求最高速度，但显存余量较小。运行时应尽量关闭 NVIDIA Broadcast、Wallpaper Engine、视频应用、GPU Overlay 和重度浏览器页面。

---

## 13. 真实视觉请求测试

### 13.1 测试图片

```text
C:\Windows\Web\4K\Wallpaper\Windows\img0_1920x1200.jpg
```

文件大小：

```text
542091 bytes
```

分辨率：

```text
1920 × 1200
```

### 13.2 请求结构

接口：

```text
POST http://127.0.0.1:8081/v1/chat/completions
```

文本提示：

```text
Describe this image concisely, then list three visible details.
```

图片通过：

```text
data:image/jpeg;base64,...
```

发送。

生成参数：

```json
{
  "max_tokens": 128,
  "temperature": 0.2,
  "stream": false
}
```

### 13.3 默认视觉 token 测试

参数中没有显式限制：

```text
--image-min-tokens
--image-max-tokens
```

程序警告：

```text
Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
```

结果：

| 指标 | 数据 |
|---|---:|
| 图片/提示处理 token | 2305 |
| 日志中的视觉处理 token | 2301 |
| 图片提示处理时间 | 34294.09 ms |
| 图片提示处理速度 | 67.21 t/s |
| 生成 token | 128 |
| 生成时间 | 1992.13 ms |
| 生成速度 | 64.25 t/s |
| 总时间 | 36286.21 ms |
| 请求墙钟时间 | 约 36.4 秒 |

中间生成日志：

```text
n_decoded = 100, tg = 64.17 t/s
```

主要问题不是文本生成慢，而是单张图片被编码成约 2300 个视觉 token，首轮图片处理耗时约 34.3 秒。

### 13.4 将视觉 token 固定为 1024

新增：

```text
--image-min-tokens 1024
--image-max-tokens 1024
```

结果：

| 指标 | 数据 |
|---|---:|
| 图片/提示处理 token | 1025 |
| 日志中的视觉处理 token | 1021 |
| 图片提示处理时间 | 9608.99 ms |
| 图片提示处理速度 | 106.67 t/s |
| 生成 token | 128 |
| 生成时间 | 1956.53 ms |
| 生成速度 | 65.42 t/s |
| 总时间 | 11565.52 ms |
| 请求墙钟时间 | 约 11.65 秒 |

中间生成日志：

```text
n_decoded = 100, tg = 65.38 t/s
```

### 13.5 视觉 token 优化收益

端到端时间：

```text
36.4 秒 → 11.65 秒
减少约 24.75 秒
降低约 68.0%
约为原来的 3.12 倍速度
```

视觉/提示处理速度：

```text
67.21 t/s → 106.67 t/s
提高约 58.7%
```

文本生成速度：

```text
64.25 t/s → 65.42 t/s
基本不变，约提高 1.8%
```

结论：

- 图片首轮延迟的主要决定因素是视觉 token 数量。
- 限制到 1024 后，文本生成速度没有受损。
- 1024 是程序日志为 Qwen-VL grounding 建议的最低值。
- 普通图片描述、物体识别和场景理解偏向速度时适合 1024。
- 极小文字 OCR、密集图表、超高分辨率细节和精确定位任务可能受精度影响。

---

## 14. 最终 `start.bat`

最终写入：

```bat
@echo off
chcp 65001 >nul
cd /d "%~dp0"

llama-server.exe ^
-m "models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf" ^
--mmproj "models\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf" ^
--mmproj-offload ^
--image-min-tokens 1024 ^
--image-max-tokens 1024 ^
-ngl all ^
--fit off ^
--flash-attn on ^
--cache-type-k q8_0 ^
--cache-type-v q8_0 ^
-c 8192 ^
-n 4096 ^
-np 1 ^
--host 127.0.0.1 ^
--port 8080

pause
```

### 14.1 参数逐项说明

| 参数 | 作用 | 选择理由 |
|---|---|---|
| `--mmproj` | 加载视觉投影模型 | 保留图片/视频理解能力 |
| `--mmproj-offload` | 将 mmproj 计算卸载到 GPU | 用户要求视觉模型必须用 GPU |
| `--image-min-tokens 1024` | 单图最少视觉 token | 满足日志建议的 Qwen-VL grounding 最低值 |
| `--image-max-tokens 1024` | 单图最多视觉 token | 将测试图首轮耗时从 36.4 秒降至 11.65 秒 |
| `-ngl all` | 尽可能将全部主模型层放到 GPU | 长测比自动适配快约 23.6% |
| `--fit off` | 禁用自动显存适配 | 避免自动减少 GPU 层及 `-ngl` 冲突警告 |
| `--flash-attn on` | 启用 Flash Attention | 降低注意力计算/显存成本 |
| `--cache-type-k q8_0` | K KV Cache 使用 Q8 | 实测比 Q4 更快 |
| `--cache-type-v q8_0` | V KV Cache 使用 Q8 | 实测比 Q4 更快 |
| `-c 8192` | 总上下文 8192 | 比 4096 提供更大可用上下文，长测速度未下降 |
| `-n 4096` | 单次最多生成 4096 token | 保留长回复能力；实际 API 参数仍可覆盖 |
| `-np 1` | 只开一个服务 slot | 防止默认 4 slot 增加 KV/上下文压力 |
| `--host 127.0.0.1` | 仅本机访问 | 避免服务暴露到局域网 |
| `--port 8080` | 服务端口 | 保持原有客户端兼容 |

---

## 15. 最终结果总览

| 阶段 | 后端/配置 | 结果 |
|---|---|---:|
| CUDA 依赖缺失 | CPU，`llama-bench tg128` | 11.65 t/s |
| CUDA 初次启用 | CUDA，短 `llama-bench tg64` | 19.01 t/s |
| 显存顶满 + 自动 4 slot | 服务端真实生成 | 约 9.46 → 8.91 t/s |
| 自动适配 256 MiB，8K，1 slot | 384-token API 长测 | 53.59 t/s |
| 强制全层，8K，1 slot | 384-token API 长测 | 66.25 t/s |
| 最终配置，默认图片 token | 图片后生成 | 64.25 t/s |
| 最终配置，图片 token=1024 | 图片后生成 | 65.42 t/s |
| 默认视觉 token | 单图总时间 | 36.4 s |
| 视觉 token=1024 | 单图总时间 | 11.65 s |

注意：`llama-bench`、服务端 API、短生成、长生成和视觉请求不是同一种负载，数字不能简单横向等同比较。最终应以真实服务端 API 长测和真实图片请求为主。

---

## 16. 已知风险与限制

### 16.1 显存余量很小

最快组合测试时整卡显存使用：

```text
11873 / 12288 MiB
```

理论剩余约：

```text
415 MiB
```

因此以下程序可能导致速度下降、VMM 分页或 OOM：

- NVIDIA Broadcast
- Wallpaper Engine
- NVIDIA Overlay
- 视频播放/直播应用
- 浏览器 GPU 加速页面
- 其他 CUDA/AI 程序

### 16.2 强制全层存在环境敏感性

`all-q8-ctx4096` 曾在一次短测中只有 16.37 t/s，而 `all-q8-ctx8192` 在后续长测稳定约 66.25 t/s。

这说明 `-ngl all --fit off` 在显存贴边时可能受以下因素影响：

- 启动时可用显存
- Windows WDDM/VMM 状态
- 后台 GPU 程序
- GPU 时钟/功耗状态
- 先前进程退出后的内存回收
- CUDA 图和 kernel 预热

若未来再次出现低于约 50 t/s 的稳定生成速度，应先关闭后台 GPU 应用并重启服务。

### 16.3 1024 视觉 token 的质量取舍

1024 偏向低延迟。以下任务可能需要提高到 1536、2048 或恢复模型默认：

- 小字号 OCR
- 密集表格
- 复杂图表
- 精确物体定位
- 远距离小目标
- 超高分辨率截图

提高视觉 token 会显著增加首轮图片处理时间，但通常不会明显改变后续文本生成速度。

### 16.4 上下文长度

模型训练上下文日志为：

```text
n_ctx_train = 262144
```

当前只使用：

```text
n_ctx_seq = 8192
```

日志警告：

```text
n_ctx_seq (8192) < n_ctx_train (262144)
the full capacity of the model will not be utilized
```

这是提示而不是错误。RTX 3060 12GB 无法在当前模型、F16 mmproj、Q8 KV 和全 GPU 权重的条件下合理使用 262K 上下文。

---

## 17. 故障排查与复验命令

### 17.1 确认 CUDA 设备

```powershell
.\llama-server.exe --list-devices
```

预期：

```text
CUDA0: NVIDIA GeForce RTX 3060
```

### 17.2 查看 GPU 状态

```powershell
nvidia-smi
```

或：

```powershell
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,pstate,clocks.current.sm,power.draw --format=csv,noheader
```

### 17.3 查看服务 slot

```powershell
Invoke-RestMethod http://127.0.0.1:8080/slots | ConvertTo-Json -Depth 8
```

预期：

```text
Count = 1
n_ctx = 8192
```

### 17.4 查看服务属性

```powershell
Invoke-RestMethod http://127.0.0.1:8080/props | ConvertTo-Json -Depth 8
```

应确认：

```text
modalities.vision = true
total_slots = 1
```

### 17.5 性能日志重点

关注：

```text
prompt eval time
eval time
tg = ... t/s
```

生成速度以最终：

```text
eval time = ... / N tokens (... tokens per second)
```

为主要参考。

---

## 18. 回滚方法

原配置备份：

```text
G:\llama-b9672-bin-win-cuda-13.3-x64\start.bat.before-tuning.bak
```

PowerShell 回滚命令：

```powershell
Copy-Item `
  -LiteralPath "G:\llama-b9672-bin-win-cuda-13.3-x64\start.bat.before-tuning.bak" `
  -Destination "G:\llama-b9672-bin-win-cuda-13.3-x64\start.bat" `
  -Force
```

### 更保守的显存配置

若最快配置因后台 GPU 占用而不稳定，可改为：

```bat
-ngl auto ^
-fitt 256 ^
--flash-attn on ^
--cache-type-k q8_0 ^
--cache-type-v q8_0 ^
-c 8192 ^
-np 1 ^
```

该配置长测约：

```text
53.59 t/s
```

比最快配置慢约 19%，但显存余量更大，启动环境适应性更好。

---

## 19. 最终结论

1. 最初约 12 token/s 的根因是 CUDA 13.3 运行依赖缺失，实际使用 CPU；CPU 基准为 11.65 token/s。
2. 安装匹配的 `cublas64_13.dll`、`cublasLt64_13.dll` 和 `cudart64_13.dll` 后，CUDA0 成功识别。
3. CUDA 启用后曾出现约 9 token/s，是因为 16K 上下文、F16 mmproj、Q8 KV、默认 4 slot 和其他 GPU 程序共同将显存推到 11950 MiB，只剩 161 MiB。
4. `-np 1` 是服务端性能恢复的关键参数之一。
5. Q4 KV 在本机实测慢于 Q8，因此最终使用 Q8/Q8。
6. 自动适配 `-ngl auto -fitt 256` 稳定约 53.59 token/s。
7. 强制全层 `-ngl all --fit off`、8K 上下文、单 slot 的长测达到 66.25 token/s，是本次测试最快配置。
8. 真实图片请求证明主模型和视觉路径均在 CUDA 环境下工作。
9. 将单图视觉 token 固定为 1024 后，1920×1200 测试图的端到端时间从约 36.4 秒下降到 11.65 秒，同时图片后文本生成保持约 65.42 token/s。
10. 最终配置以“12GB 显存下的最高速度”为目标，显存余量较小；后台 GPU 占用较高时应切换到自动适配保守配置。

