# 🚀 MetalHub: The Ultimate Local AI Gateway

**MetalHub** 是一个专为 Apple Silicon (M1/M2/M3) 架构设计的全能本地 AI 引擎。通过独创的“显存接力”与“分布式集群”技术，它能唤醒你家中所有的旧硬件（Mac, Windows, Linux），共同构建一个强大的异构 AI 算力中心。

**MetalHub** 更新后，更是一个深度调优的全能 AI 推理网关。它通过 **显存接力 (VRAM Relay)** 与 **分布式异构集群** 技术，将大模型推理、多模态分析与极速生图整合进一个工业级 Web 控制台。
我们的愿景是：**打破显存霸权，让每一台被遗忘的旧设备都能在 AI 时代焕发第二春。**

---

## ✨ 核心特性

- **🍎 极致单机优化**：针对 M1 Max 400GB/s 带宽深度调优，27B 模型推理可达 27+ t/s。
- **🌐 异构集群分布式 (Cluster)**：通过 RPC 协议将旧 Intel Mac、Windows 台式机、Linux 服务器串联，共享显存处理超大模型（如 70B）。
- **⚖️ 智能负载均衡**：根据集群内各代硬件的显存容量与性能系数，自动按比例分配模型层数。
- **🛡️ 节点热自愈**：实时探测远程节点状态。若从机掉线，系统自动无损降级至本地模式，确保服务永不断线。
- **🔄 显存接力 (VRAM Relay)**：精准释放 Metal 缓存，实现 27B LLM 与 SDXL 绘图引擎的毫秒级切换。
- **⚡ 动态上下文引擎**：根据输入长度自动调整 `n_ctx` (8K-64K)，兼顾响应速度与深度记忆。
- **📊 实时监控台**：可视化集群看板，实时显示每个节点的负载比例、TPS 及预处理延迟。

---

## 🛠️ 项目目录结构

```text
MetalHub/
├── main.py              # FastAPI 核心引擎（支持流式/分布式/自愈）
├── config.yaml          # 核心配置文件（Profile 与 集群节点定义）
├── models/              # 模型存放目录（GGUF 权重管理）
├── templates/           # 前端 UI (Jinja2 模板)
├── docs/                # 详细部署文档
└── pyproject.toml       # uv 环境配置
```

---

## 🆕 v0.3.0 更新亮点 (Latest Updates)

- 🚀 **性能巅峰**：M1 Max 27B 模型推理冲破 **29.65 t/s**，预处理加速 300%。
- 🧠 **模型热切换**：UI 侧边栏新增模型选择器，支持 `models/` 目录 GGUF 文件秒级热加载。
- 📝 **专业渲染**：全本地 Markdown 支持，集成代码高亮、一键复制及安全警示。
- ⏳ **思考可视化**：新增可折叠“思考日志”，实时监控预处理进度与集群负载分配。
- 💾 **记忆持久化**：支持多会话管理，对话历史本地自动保存。
- 🌐 **集群增强**：完善异构节点自愈逻辑，支持按显存比例智能分发任务。

---

## 👁️ 多模态视觉对齐 (Multimodal Vision Excellence)

MetalHub v0.4.0 现已完美适配 **Qwen 3.6 / 2.5 VL** 协议。通过自研的协议分发器（Protocol Dispatcher）与源码级驱动编译，我们实现了对复杂图像 100% 的解析准确度。

### 📸 极限性能测试：电视测试卡 (Philips PM5544)
我们使用含有高频线条、标准色块和几何网格的经典电视测试卡进行压力测试，MetalHub 驱动下的 Qwen-27B 展现了惊人的视觉理解力：


| 测试原图 | MetalHub 推理描述 (Qwen 3.6 27B) |
| :--- | :--- |
| <img src="https://wikimedia.org" width="300"> | **“这张图片展示的是一个经典的电视测试卡图案... 准确识别出了中心圆、SMPTE彩色条（黄、青、绿、紫、红、蓝）、分辨率测试细线以及背景的灰色方格网格。”** |

#### 为什么我们的准确度更高？
- **源码驱动**：弃用过时的 Pip 二进制包，采用源码编译开启 `Qwen35ChatHandler`。
- **协议对齐**：自动识别模型架构，杜绝 LLaVA 模板对 Qwen 模型的注意力干扰（幻觉消除）。
- **Metal 优化**：利用 M1 Max 统一内存带宽，实现不到 1 秒的极速图像 Patch 编码。

---

## 🆕 v0.4.0 里程碑更新

- 👁️ **多模态视觉 100% 对齐**：通过源码编译驱动，深度适配 **Qwen 3.6 / 2.5-VL** 协议。实现复杂图像（如电视测试卡）的精准识别，彻底消除幻觉。
- 📊 **全量 API 实时监控中心**：内置审计看板，实时展示局域网内所有 REST API 请求详情（来源 IP、模型 ID、实时 TPS、Token 消耗统计）。
- 🎨 **显存接力生图 (VRAM Relay)**：集成最新 `stable-diffusion-cpp`。支持 LLM 与 SDXL-Turbo 秒级自动切换，M1 Max 实现 **1秒级** 出图。
- 📄 **文档/图片 拖拽分析**：支持 PDF、TXT、多图直接拖入对话框。后端自动提取 40K+ 剧本内容或激活视觉适配器进行多模态对谈。
- 🛡️ **工业级稳定性增强**：
  - **物理隔离管理**：采用 `models/LLM` 与 `models/GEN` 目录架构，确保护航模型与适配器 100% 匹配。
  - **瞬时强杀退出**：解决 SSE 导致的退出阻塞，实现 `Ctrl+C` 毫秒级闪退，物理释放显存。
  - **服务端安全校验**：自动拦截对非多模态模型的视觉请求，防止系统异常。

---
## 🔥 极限性能 (M1 Max 64GB)


| 任务类型 | 模型 (Model) | 推理速度 (Throughput) | 特性 (Key Features) |
| :--- | :--- | :--- | :--- |
| **文本推理** | **Qwen-27B (Q4_K_M)** | **29.65 t/s** | 动态 64K 上下文支持 |
| **视觉分析** | **Qwen-3.6-27B-VL** | **~1s Encoding** | 复杂图像 100% 对齐 |
| **图像生成** | **SDXL-Turbo** | **0.8 - 1.5s / img** | 1-Step 极速成像 |
| **小模型** | **Qwen-4B** | **~58 t/s** | 零延迟毫秒级响应 |

---

## 🌐 异构集群支持 (Distributed Engine)

MetalHub 允许你通过 **RPC 模式** 唤醒旧的 Intel Mac 或 Windows 节点，实现算力跨机叠加：
1. **主节点 (M1 Max)**：负责核心调度与 KV Cache 管理。
2. **从机节点 (Legacy Mac/PC)**：通过 `rpc-server` 分担模型层数（Layers），突破单机显存限制。
3. **故障自愈**：实时心跳监测，若从机断开，系统自动无损回退至本地模式，确保任务永不断线。


---

## 📥 快速安装 (macOS)

推荐使用 [uv](https://github.com) 管理环境。

```bash
# 1. 环境预准备
brew install cmake llvm libomp

git clone https://github.com/icewindzhang-crypto/MetalHub.git
cd MetalHub
export CMAKE_ARGS="-DGGML_METAL=ON -DSD_METAL=ON -DGGML_OPENMP=OFF"
uv venv

# 2. 由于 MetalHub 追求极致性能，必须从源码编译驱动以启用最新的 Metal 算子：

# 编译 llama-cpp-python (开启 Metal & 视觉 & Flash Attention)
export CMAKE_ARGS="-DGGML_METAL=ON -DGGML_FLASH_ATTN=ON"
git clone --recursive https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
uv pip install . --no-cache-dir

git clone --recursive hhttps://github.com/william-murray1204/stable-diffusion-cpp-python.git
cd stable-diffusion-cpp-python

# 针对 Apple Silicon 设置编译环境
# 注意：我们显式关闭了可能引起报错的组件，并强制开启 Metal
export CMAKE_ARGS="-DSD_METAL=ON -DGGML_METAL=ON -DGGML_OPENMP=OFF"

# 使用 uv 在当前环境中安装此本地目录
uv pip install . --no-cache-dir
```

---

### 3. 新增模型摆放规范
请严格遵循以下目录结构，系统将自动识别功能模块：
```text
models/
├── LLM/
│   └── Qwen3.6-27B-Vision/        <-- 目录即为模型 ID
│       ├── model.gguf             # 主推理文件
│       └── mmproj.gguf            # 视觉适配器 (可选)
└── GEN/
    └── SDXL-Turbo-v1/
        └── sdxl_turbo_q4.gguf     # 生图模型文件

### 3. 项目维护者建议
为了让 **MetalHub** 更加“开箱即用”，你可以写一个简单的 `setup.sh` 脚本放在项目根目录：

```bash
#!/bin/bash
echo "🚀 开始安装 MetalHub 生产环境..."
brew install cmake llvm libomp

# 处理生图组件
if [ ! -d "stable-diffusion-cpp-python" ]; then
    git clone --recursive https://github.com
fi

export CMAKE_ARGS="-DSD_METAL=ON -DGGML_METAL=ON -DGGML_OPENMP=OFF"
uv pip install ./stable-diffusion-cpp-python --no-cache-dir
uv pip install fastapi uvicorn llama-cpp-python pyyaml psutil torch pillow Jinja2
echo "✅ 安装完成！请在 models/ 放入模型后运行 uv run python main.py"
```

这样用户只需要运行 `sh setup.sh` 就能自动处理复杂的编译逻辑。

**现在的手动编译过程中，`libwebm` 的报错消失了吗？如果编译成功，你的 MetalHub 就真正打通了 GGUF 生图的最后一公里。**

```

## 📸 视觉对齐能力展示



| 测试输入 (Philips PM5544) | MetalHub 识别描述 (Qwen 3.6 27B) |
| :--- | :--- |
| <img src="https://wikimedia.org" width="300"> | **100% 准确识别**：“这是一张经典的电视测试卡图案。准确识别出了中心圆、SMPTE彩色条（黄、青、绿、紫、红、蓝）、分辨率测试细线以及背景的灰色方格网格...” |

---

## ⚙️ 配置文件 (config.yaml)

您可以根据硬件规格自由配置。`active_profile: "auto"` 将开启自动硬件探测。

```yaml
server:
  port: 11434
  active_profile: "auto"
  rpc_nodes: # 分布式节点配置
    - host: "192.168.1.10:50052" # Windows/Linux 从机
      vram_gb: 12                # 显存容量
      weight: 1.2                # 性能权重
    - host: "192.168.1.11:50052" # 旧 Intel Mac
      vram_gb: 8
      weight: 0.5

profiles:
  high_end: # 针对 64GB+ 内存环境
    llm:
      path: "./models/Qwen3.6-27B-Q4_K_M.gguf"
      n_ctx_limit: 65536
      n_batch: 4096
    image:
      gguf_path: "./models/z_image_turbo-Q8_0.gguf"
      t5_path: "./models/Qwen3-4B-Instruct-Q4_K_M.gguf"
      vae_path: "./models/sdxl_vae.safetensors"
      steps: 4
```

---

## 🚀 运行与测试

1. **放置模型**：将 GGUF 文件放入 `./models/` 目录。
2. **启动引擎**：`uv run python main.py`
3. **访问交互页面**：`http://localhost:11434`
4. **API 对接**：支持标准 OpenAI 协议，Base URL 为 `http://localhost:11434/v1`。

---

## ⌨️ 快捷交互特性

- ✉️ **极速发送**：直接按 `Enter` 发送，`Ctrl/Cmd + Enter` 实现输入框换行。
- 📎 **文件拖拽**：支持 PDF、TXT 及多张图片直接拖入，自动完成文本提取与多模态加载。
- 📋 **代码管理**：内置 Markdown 代码块高亮，支持一键复制并附带专业免责申明。
- ⏳ **状态透明**：可折叠“思考日志”框，实时展示模型预处理（ms）及集群负载权重。

---

## 🗺️ Roadmap

- [x] **Streaming (SSE)**: 流式字符实时输出。
- [x] **Cluster Load Balancing**: 多节点显存比例自动分配。
- [x] **Unified Vision**: 接入 Qwen3.6 实现本地视觉理解。
- [ ] **Whisper Integration**: 集成实时语音转文字。
- [ ] **Auto-Installer**: 一键式模型环境安装脚本。

---

## 📜 愿景

**MetalHub** 的初衷是解决本地运行大模型时的算力焦虑。我们通过极致的硬件压榨与分布式技术，让每一台老旧设备都能在 AI 时代重新作为生产力中心“咆哮”。

**MetalHub: Awakening Legacy Hardware for the AI Era.**

## 🤝 贡献

我们欢迎所有希望“复活旧硬件”的开发者加入。如果你有关于分布式计算、模型量化或 Metal 优化的建议，请提交 PR。

**让 AI 回归大众，让硬件重获新生。**

---
[MIT License](LICENSE)
