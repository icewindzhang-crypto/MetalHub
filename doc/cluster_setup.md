# 🌐 MetalHub 异构集群分布式部署指南 (Cluster Setup)

MetalHub 支持利用 `llama.cpp` 的 RPC 机制，将局域网内的旧款 Mac、Windows 台式机或 Linux 服务器转化为计算卫星，共同分担大模型的推理压力。

---

## 🛠️ 1. 从机 (Worker) 部署

在你想“唤醒”的旧设备上，根据系统选择以下步骤：

### 🪟 Windows 端 (推荐带旧 NVIDIA 显卡)
1. **安装环境**：安装 [CMake](https://cmake.org) 和 [CUDA Toolkit](https://nvidia.com)。
2. **编译工具**：
   ```powershell
   git clone https://github.com
   cd llama.cpp
   mkdir build; cd build
   cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON 
   cmake --build . --config Release
   ```
3. **启动服务**：
   ```powershell
   .\(\bin\Release\rpc-\)server.exe -H 0.0.0.0 -p 50052
   ```

### 🐧 Linux 端 (推荐旧服务器/办公机)
1. **安装依赖**：`sudo apt install build-essential cmake`
2. **编译工具**：
   ```bash
   git clone https://github.com
   cd llama.cpp
   mkdir build && cd build
   cmake .. -DGGML_RPC=ON
   make -j
   ```
3. **启动服务**：
   ```bash
   ./bin/rpc-server -H 0.0.0.0 -p 50052
   ```

### 🍎 Intel Mac 端 (2017-2020 款)
1. **编译工具**：
   ```bash
   cmake .. -DGGML_METAL=ON -DGGML_RPC=ON
   cmake --build . --config Release
   ```
2. **启动服务**：
   ```bash
   ./bin/rpc-server -H 0.0.0.0 -p 50052
   ```

---

## 🧠 2. 主机 (Master) 配置：MetalHub 容错优化

为了确保集群的稳定性，MetalHub 引入了 **“节点热探测与自动退避”** 机制。

### 修改 `config.yaml`
在服务器配置中增加远程节点列表：
```yaml
server:
  port: 11434
  active_profile: "auto"
  # 添加局域网内所有可用的从机 IP
  rpc_nodes: 
    - "192.168.1.10:50052" # Windows 台式机
    - "192.168.1.11:50052" # 旧 Intel Mac
```

### 核心自愈逻辑 (已集成至 main.py)
MetalHub 在每次模型加载前会执行以下逻辑：
1. **心跳检测**：通过 Socket 尝试连接 `rpc_nodes` 中的地址。
2. **动态过滤**：仅将响应成功的节点传递给推理引擎。
3. **无损降级**：若所有远程节点均掉线，系统自动回退至 M1 Max 本地运行，确保前端服务不断联。

---

## ⚠️ 集群运行注意事项

1. **网络连接**：强烈建议所有节点通过 **千兆有线网 (Ethernet)** 连接。WiFi 延迟会显著拉低推理的 t/s。
2. **版本对齐**：所有从机的 `rpc-server` 版本必须与主机的 `llama-cpp-python` 内核版本保持一致（建议同时更新）。
3. **显存接力**：分布式模式下，生图任务（SD-CPP）目前主要在本地运行，建议在 `config.yaml` 中为生图预留足够的本地显存空间。

---

> **“让每一台被遗忘的机器，都成为 MetalHub 神经网络的一部分。”**

