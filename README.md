# 🚀 MetalHub

**MetalHub** 是一个专为 Apple Silicon (M1/M2/M3) 架构设计的全能本地 AI 引擎。通过独创的“显存接力”与“分布式集群”技术，它能唤醒你家中所有的旧硬件（Mac, Windows, Linux），共同构建一个强大的异构 AI 算力中心。

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

## 📥 快速安装 (macOS)

推荐使用 [uv](https://github.com) 管理环境。

```bash
# 1. 环境预准备
brew install cmake llvm libomp

# 2. 克隆并安装 (强制开启 Metal 加速并解决 OpenMP 冲突)
git clone https://github.com
cd MetalHub
export CMAKE_ARGS="-DGGML_METAL=ON -DSD_METAL=ON -DGGML_OPENMP=OFF"
uv venv

### 2. 深度安装 (解决 libwebm 编译报错)

由于 `stable-diffusion.cpp` 包含复杂的子模块（如 libwebm），直接通过 `uv add` 可能会失败。请按以下步骤手动编译：

```bash
# 克隆并拉取完整子模块
git clone --recursive https://github.com
cd stable-diffusion-cpp-python

# 针对 Apple Silicon 设置编译环境
# 注意：我们显式关闭了可能引起报错的组件，并强制开启 Metal
export CMAKE_ARGS="-DSD_METAL=ON -DGGML_METAL=ON -DGGML_OPENMP=OFF"

# 使用 uv 在当前环境中安装此本地目录
uv pip install . --no-cache-dir
```

---

### 2. 为什么这样做能解决问题？
1. **`--recursive`**: 这是关键。它会把 `vendor/stable-diffusion.cpp/thirdparty/libwebm` 里的所有 `.cmake` 配置文件完整拉取下来，解决你遇到的 `include could not find requested file` 报错。
2. **`--no-cache-dir`**: 强制重新编译。你之前的失败尝试可能会在 `uv` 缓存中留下半成品，这个参数确保从零开始构建。
3. **`-DGGML_OPENMP=OFF`**: 继续保持关闭 OpenMP，以适配 macOS 原生 Clang。

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

## 🗺️ Roadmap

- [x] **Streaming (SSE)**: 流式字符实时输出。
- [x] **Cluster Load Balancing**: 多节点显存比例自动分配。
- [ ] **Unified Vision**: 接入 Qwen2-VL 实现本地视觉理解。
- [ ] **Whisper Integration**: 集成实时语音转文字。
- [ ] **Auto-Installer**: 一键式模型环境安装脚本。

---

## 🤝 贡献

我们欢迎所有希望“复活旧硬件”的开发者加入。如果你有关于分布式计算、模型量化或 Metal 优化的建议，请提交 PR。

**让 AI 回归大众，让硬件重获新生。**

---
[MIT License](LICENSE)
