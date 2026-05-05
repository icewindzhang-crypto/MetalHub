# 🚀 MetalHub

**MetalHub** 是一个专为 Apple Silicon (M1/M2/M3) 架构设计的全能本地 AI 引擎。它通过独创的“显存接力”技术，实现了在大参数语言模型（LLM）与高质量生图模型（SDXL）之间的秒级切换，让你的 Mac 成为真正的全能 AI 工作站。

---

## ✨ 核心特性

- **🍎 深度优化**：基于 Metal 框架编译，原生压榨 M1/M2/M3 Max 内存带宽性能。
- **🔄 显存接力 (VRAM Relay)**：100% 物理释放 Metal 缓存，确保 27B 规模 LLM 与 SDXL Turbo 模型在同一台设备上串行流转，互不干扰。
- **🧠 智能预设 (Auto-Profile)**：启动时自动感知统一内存大小，智能推荐并加载最适合当前硬件的模型参数。
- **⚡ 动态上下文**：根据输入文本长度实时调整 `n_ctx`。短文本秒开，长文本（支持 40K+）不丢记忆。
- **🎨 兼容性**：完美适配 OpenAI API 协议（包含 `/v1/chat/completions` 流式输出与 `/v1/images/generations`）。
- **📊 实时监控**：自带监控控制台，实时显示 TPS（生成速度）、Processing（预处理延迟）及内存状态。

---

## 🛠️ 环境准备

### 1. 硬件要求
- **推荐**：Apple M-系列芯片 (M1 Max / M2 Ultra 等)，64GB 或以上统一内存。
- **最低**：8GB 内存 (自动进入 `low_end` 模式)。

### 2. 系统依赖
在安装 Python 依赖前，请确保系统已安装 `cmake` 和 `llvm`：
```bash
brew install cmake llvm libomp
```

---

## 📥 快速安装

我们推荐使用 [uv](https://github.com) 进行高效的依赖管理。

```bash
# 1. 克隆项目
git clone https://github.com
cd MetalHub

# 2. 初始化环境并从源码编译 (开启 Metal 加速)
# 注意：我们关闭了 OpenMP 以避免 macOS 上的编译冲突，性能由 Metal 保证
export CMAKE_ARGS="-DGGML_METAL=ON -DSD_METAL=ON -DGGML_OPENMP=OFF"
uv venv
uv pip install fastapi uvicorn llama-cpp-python stable-diffusion-cpp-python pyyaml psutil torch pillow Jinja2
```

---

## 📂 模型管理

请将下载好的 GGUF 模型放入 `./models/` 目录下。

### 推荐组合：
- **LLM**: [Qwen3.6-27B-Instruct-GGUF](https://huggingface.co)
- **Image**: [Z-Image-Turbo-GGUF](https://huggingface.co)
- **Text Encoder**: [Qwen3-4B-Instruct-GGUF](https://huggingface.co)
- **VAE**: [sdxl_vae.safetensors](https://huggingface.co)

---

## ⚙️ 配置说明

编辑项目根目录下的 `config.yaml`：
```yaml
server:
  port: 11434
  active_profile: "auto" # 可选 "auto", "high_end", "mid_range", "low_end"

profiles:
  high_end:
    llm:
      path: "./models/Qwen3.6-27B-Q4_K_M.gguf"
      n_ctx_limit: 65536
      n_batch: 4096 # M1 Max 400GB/s 带宽优化
    image:
      gguf_path: "./models/z_image_turbo-Q8_0.gguf"
      t5_path: "./models/Qwen3-4B-Instruct-Q4_K_M.gguf"
      vae_path: "./models/sdxl_vae.safetensors"
      steps: 4
      cfg_scale: 1.0
```

---

## 🚀 启动与使用

### 1. 运行引擎
```bash
uv run python main.py
```

### 2. 交互测试
打开浏览器访问：`http://localhost:11434`

### 3. API 对接
你可以将 **Claude Code**, **LobeChat**, **NextChat** 等客户端的 API Base 指向：
`http://localhost:11434/v1`

---

## 🗺️ Roadmap (近期开发计划)
- [ ] **Streaming (SSE)**: 实现更流畅的流式字符输出响应。 (正在进行中)
- [ ] **Multi-Modal**: 集成 Whisper 语音识别与视觉理解能力。
- [ ] **Task Queue**: 引入请求队列，支持更稳健的并发管理。
- [ ] **Auto-Downloader**: 增加模型一键下载与校验功能。
- [ ] **GUI Wrapper**: 为非开发者提供更友好的图形启动界面。


## 🤝 贡献与反馈
本项目目前处于 Alpha 阶段。欢迎提交 Issue 或 Pull Request，一起构建 Apple Silicon 上最强的 AI 枢纽。

---

## ⚖️ 开源协议
MIT License

