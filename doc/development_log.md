# 🛠️ MetalHub 研发日志 (Development Log)

这是一份记录了 MetalHub 从 0 到 1 诞生过程的日志，涵盖了针对 Apple Silicon (M1 Max) 的极限性能调优与架构演进方案。

---

## 📅 阶段一：性能破局 (Performance Breakthrough)

### 核心挑战
- **初始困境**：Qwen 3.6 27B 模型在 128K 超长上下文模式下，预处理（Prompt Processing）极其缓慢，生成速度仅为 **7.6 t/s**，且 CPU 占用极高，GPU 频繁跳水。
- **瓶颈诊断**：
    - 上下文预分配过大（128K）导致内存带宽负载过重。
    - `llama-server` 默认线程调度包含了能效核（E-cores），拖慢了性能核（P-cores）的同步。
    - KV Cache 量化在长文本预处理阶段存在计算回退。

### 调优成果
- **最终参数**：`-b 4096`, `-ub 1024`, `-t 8`, `--flash-attn`。
- **性能飞跃**：在 40K 长文本预处理时间从 11 分钟缩短至 **2 分钟内**；短文本生成速度稳定在 **27.63 t/s**。
- **关键结论**：在 M1 Max 400GB/s 带宽下，8 个性能核是推理的最优解；Batch Size 设为 4096 能极大压榨 GPU 的吞吐能力。

---

## 📅 阶段二：显存接力架构 (VRAM Relay Architecture)

### 设计灵感
- **痛点**：用户希望在 64GB 内存上同时运行 27B 大模型（约 30GB）和 SDXL 级别的生图模型（约 15GB）。
- **解决方案**：引入 **“串行接力”** 机制。通过 FastAPI 网关对 API 请求进行拦截，确保 LLM 推理完成后，通过强制销毁对象（`del`）和垃圾回收（`gc.collect()`, `torch.mps.empty_cache()`）物理清空显存，再加载生图模型。
- **意义**：实现了在有限的 64GB 统一内存中，无损运行超大规模图文组合。

---

## 📅 阶段三：动态调优引擎 (Dynamic Tuning Engine)

### 创新功能
- **动态上下文 (Dynamic n_ctx)**：
    - 实现 `get_optimized_params` 函数。根据输入文本长度，在 8K、32K、64K 之间自动切换。
    - **逻辑**：小文本分配小空间以换取极速响应；长文本自动扩展以保证记忆完整。
- **自动硬件适配**：
    - 程序启动时自动通过 `psutil` 检测物理内存总量，智能选择 `low_end`, `mid_range` 或 `high_end` 配置文件。

---

## 📅 阶段四：商业化演进 (Commercial Evolution)

### 目录结构化
- **解耦设计**：将核心逻辑 (`main.py`)、用户配置 (`config.yaml`)、交互模板 (`templates/`) 和资源目录 (`models/`) 彻底分离。
- **OpenAI 全兼容**：不仅支持文本生成，还通过 GGUF 格式打通了 `/v1/images/generations` 接口。

### 交互进化
- **SSE 流式输出**：实现服务端发送事件（Server-Sent Events），解决了长文本生成的“假死”感。
- **实时指标显示**：在前端 UI 实时反馈 TPS (Tokens Per Second) 和 Processing 延迟，让性能看得见。

---

## 🧪 技术栈沉淀
- **Runtime**: `llama-cpp-python` (Metal-compiled), `stable-diffusion-cpp-python`
- **Frontend**: FastAPI + Tailwind CSS + Vanilla JS (SSE support)
- **Environment**: `uv` (Fast & reliable Python package management)
- **Hardware Target**: Apple Silicon M-Series (Unified Memory Architecture)

---

- Python 层的精准卸载：原版 llama-server 为了支持多并发，有很多 KV Cache 管理的复杂逻辑。我们通过 del llm 强制物理清空，让 GPU 始终在最干净的状态下只处理当前的这一路推理。
- 去除了 Slot 竞争：llama.cpp 默认开启多 Slot，每个 Slot 都要预留显存，导致带宽被碎片化。我们通过 FastAPI 锁死为单任务，M1 Max 的 400GB/s 独宠这一个模型。
- 动态 n_batch 的魔力：手动开启了 4096 的 Batch，配合精准的 n_ctx 对齐，减少了 Metal 内存页的频繁交换。

---

> *"MetalHub 不仅仅是一个后端，它是对 Apple Silicon 极限性能的一次致敬。"*
> —— 记录于 2026 年 5 月

