import gc, os, time, torch, asyncio, base64, yaml, json, psutil, socket
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
import uvicorn
from fastapi.staticfiles import StaticFiles

import signal
import uuid

app = FastAPI()
lock = asyncio.Lock()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory="static"), name="static")

def force_exit_handler(sig, frame):
    """捕捉 Ctrl+C (SIGINT)，执行最后的物理清理并强制杀掉进程"""
    print("\n\n🛑 MetalHub 接收到关闭信号，正在执行紧急清理...")
    try:
        # 1. 尝试清空 GPU 显存（防止显存幽灵占用）
        if torch.mps.is_available():
            torch.mps.empty_cache()
        
        # 2. 物理清理：如果你的 UPLOAD_DIR 还有残留，这里可以做最后扫除
        # upload_dir = os.path.join(BASE_DIR, "uploads")
        # if os.path.exists(upload_dir):
        #     import shutil
        #     # 谨慎操作：仅删除临时上传文件，不删目录
        #     for f in os.listdir(upload_dir):
        #         os.remove(os.path.join(upload_dir, f))
                
        print("✅ 显存已释放，临时文件已清空。")
    except Exception as e:
        print(f"⚠️ 清理过程出现微小异常: {e}")
    finally:
        print("🚀 MetalHub 已完全停止。再见！")
        # 关键：使用 os._exit(0) 绕过所有 Python 层的优雅等待逻辑，直接交给内核关闭
        os._exit(0)

# 注册信号：Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, force_exit_handler)
# 注册信号：终端关闭 (SIGTERM)
signal.signal(signal.SIGTERM, force_exit_handler)


import llama_cpp.llama_chat_format as lcf

# 映射表：文件名关键字 -> (Handler类名, chat_format)
VISION_MAPPING = {
    "qwen3.6": ("Qwen35ChatHandler", "qwen3.6"),
    "qwen3.5": ("Qwen35ChatHandler", "qwen3.5"),
    "qwen2.5-vl": ("Qwen25VLChatHandler", "qwen2.5-vl"),
    "gemma4": ("Gemma4ChatHandler", "gemma4"),
    "gemma3": ("Gemma3ChatHandler", "gemma3"),
    "llama-3-vision": ("Llama3VisionAlphaChatHandler", "llama-3-vision-alpha"),
    "llava-v1.5": ("Llava15ChatHandler", "llava-1-5"),
    "llava-v1.6": ("Llava16ChatHandler", "llava-1-6"),
    "minicpm-v-2.6": ("MiniCPMv26ChatHandler", "minicpm-v-2.6"),
}

def get_vision_specs(model_filename):
    """根据文件名自动探测视觉协议"""
    fn = model_filename.lower()
    for key, (handler_name, chat_fmt) in VISION_MAPPING.items():
        if key in fn:
            return handler_name, chat_fmt
    # 默认兜底方案
    return "Llava15ChatHandler", "chatml"


import fitz  # PyMuPDF
import base64

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_uploaded_file(file_content, filename):
    ext = filename.split('.')[-1].lower()
    
    # --- 路径 A: 图像处理 (用于 Vision) ---
    if ext in ['jpg', 'jpeg', 'png', 'webp']:
         # 生成唯一文件名保存到本地，防止 Base64 撑爆内存
        save_name = f"{uuid.uuid4()}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        with open(save_path, "wb") as f:
            f.write(file_content)

        # 将图片转为 Base64 方便前端直接预览和后续发送
        base64_data = base64.b64encode(file_content).decode('utf-8')
        return {
            "type": "image",
            "mime": f"image/{ext if ext != 'jpg' else 'jpeg'}",
            "data": base64_data,
            "filename": filename,
            "local_path": save_path 
        }
    
    # --- 路径 B: 文档提取 (用于长文本) ---
    text = ""
    if ext == 'pdf':
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif ext in ['txt', 'md', 'py', 'json', 'yaml']:
        text = file_content.decode('utf-8', errors='ignore')
    
    return {
        "type": "text",
        "content": text,
        "filename": filename,
        "char_count": len(text)
    }

@app.post("/v1/files/upload")
async def upload_file(request: Request):
    form = await request.form()
    file = form.get("file")
    if not file: return {"error": "No file"}
    
    content = await file.read()
    result = process_uploaded_file(content, file.filename)
    return result



# ================= 1. 硬件探测与配置加载 
# 1. 先定义探测函数
def detect_hardware():
    total_mem = psutil.virtual_memory().total / (1024**3)
    # 稍微降低一点阈值防止临界值跳变
    if total_mem >= 48: return "high_end"
    elif total_mem >= 24: return "mid_range"
    else: return "low_end"

## 2. 确定配置文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")

# 3. 加载配置文件并赋值给全局变量 config
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 4. 现在可以使用 config 变量了
raw_name = config['server'].get('active_profile', 'auto')
if raw_name == "auto":
    active_name = detect_hardware()
else:
    active_name = raw_name

profile = config['profiles'][active_name]

# 打印调试信息，确认进入 high_end
print(f"🚀 MetalHub 模式启动: {active_name} (内存: {psutil.virtual_memory().total / (1024**3):.2f} GB)")

def check_rpc_node(host_str, timeout=2):
    """检测 RPC 节点是否在线"""
    try:
        ip, port = host_str.split(':')
        with socket.create_connection((ip, int(port)), timeout=timeout):
            return True
    except: return False

def get_balanced_rpc_config(nodes_config, local_mem_gb=48):
    """按显存比例计算分配权重"""
    online_nodes = [n for n in nodes_config if check_rpc_node(n['host'])]
    if not online_nodes: return None, [1.0]

    rpc_string = ",".join([n['host'] for n in online_nodes])
    # 计算权重：M1 Max 默认权重 1.0，其他按配置的 weight
    weights = [1.0] # 首位是本地 M1 Max
    total_score = local_mem_gb
    
    for n in online_nodes:
        score = n['vram_gb'] * n.get('weight', 1.0)
        weights.append(score)
        total_score += score
    
    # 归一化为 tensor_split 格式
    tensor_split = [w / total_score for w in weights]
    return rpc_string, tensor_split

with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

active_name = config['server'].get('active_profile', 'auto')
if active_name == 'auto': active_name = detect_hardware()
profile = config['profiles'][active_name]


import uuid, asyncio
from datetime import datetime

# ================= 审计与监控单例 =================
class AuditManager:
    def __init__(self):
        self.active_tasks = {}
        self.total_tokens_all_time = 0
        self.broadcast_queue = asyncio.Queue() # 用于将更新推送给前端监控界面

    def start_task(self, model, client_ip, task_type="TEXT"):
        task_id = str(uuid.uuid4())[:8]
        task_data = {
            "id": task_id,
            "model": model,
            "ip": client_ip,
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "status": "RUNNING",
            "type": task_type,
            "tps": 0,
            "tokens": 0
        }
        self.active_tasks[task_id] = task_data
        # 通知监控面板有新任务
        asyncio.create_task(self.broadcast_queue.put(task_data))
        return task_id

    def end_task(self, task_id, tps=0, tokens=0):
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "COMPLETED"
            self.active_tasks[task_id]["tps"] = tps
            self.active_tasks[task_id]["tokens"] = tokens
            self.total_tokens_all_time += tokens
            # 通知监控面板任务结束
            asyncio.create_task(self.broadcast_queue.put(self.active_tasks[task_id]))
            # 延迟清理（保留在界面显示一会儿）
            asyncio.create_task(self._delayed_remove(task_id))

    async def _delayed_remove(self, task_id):
        await asyncio.sleep(30) # 30秒后从内存移除
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

audit_log = AuditManager()


@app.post("/v1/system/shutdown")
async def shutdown():
    # 给前端发个信号，然后杀掉自己
    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "Shutting down..."}

# ================= 监控实时流接口 =================
@app.get("/v1/system/monitor/stream")
async def monitor_stream(request: Request):
    async def event_generator():
        # 发送初始数据
        yield f"data: {json.dumps({'type': 'INIT', 'tasks': list(audit_log.active_tasks.values()), 'total': audit_log.total_tokens_all_time})}\n\n"
        
        while True:
            # 关键：检查客户端是否已经关闭了网页或者连接已断开
            if await request.is_disconnected():
                print("📡 监控流客户端已断开")
                break
                
            try:
                # 给队列获取增加超时，每秒检查一次 is_disconnected
                update = await asyncio.wait_for(audit_log.broadcast_queue.get(), timeout=1.0)
                yield f"data: {json.dumps({'type': 'UPDATE', 'task': update, 'total': audit_log.total_tokens_all_time})}\n\n"
            except asyncio.TimeoutError:
                # 保持心跳，防止连接被中间网关切断
                yield ": ping\n\n"
                continue
            except Exception as e:
                print(f"📡 监控流异常: {e}")
                break
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")




# 修改 main.py 中的 chat_endpoint 生成器部分
# 修改 main.py 中的 stream_generator
async def stream_generator(n_ctx, n_batch, model_path):
    async with lock:
        start_time = time.time()
        # 1. 发送：正在加载模型
        yield f"data: {json.dumps({'extra': {'status': 'LOADING_MODEL', 'active_model': os.path.basename(model_path)}})}\n\n"
        
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, n_batch=n_batch, flash_attn=True, verbose=False)
        load_done = time.time()
        load_ms = round((load_done - start_time) * 1000, 2)

        # 2. 发送：正在处理 Prompt (包含预估，因为 llama-cpp-python 阻塞执行，我们发一个起始包)
        # 这里的 n_ctx 对应处理的规模
        yield f"data: {json.dumps({'extra': {
            'status': 'PROCESSING_PROMPT', 
            'processing_ms': load_ms, 
            'n_ctx': n_ctx,
            'msg': f'正在加速预处理 {n_ctx} tokens...'
        }})}\n\n"

        token_count = 0
        start_gen = time.time()
        
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            if token_count == 0:
                # 3. 发送：生成开始（此包将触发前端清除“思考”文字）
                yield f"data: {json.dumps({'extra': {'status': 'GENERATING'}})}\n\n"
            
            token_count += 1
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # 4. 发送：最终统计
        end_time = time.time()
        gen_duration = end_time - start_gen
        tps = round(token_count / gen_duration, 2) if gen_duration > 0 else 0
        yield f"data: {json.dumps({'extra': {
            'status': 'DONE',
            'tps': tps, 
            'total_tokens': token_count,
            'gen_time': round(gen_duration, 2)
        }})}\n\n"

        
        del llm
        gc.collect()
        torch.mps.empty_cache()
        
        
def scan_models(sub_dir):    
    """通用目录扫描器"""
    base_path = os.path.join(BASE_DIR, "models", sub_dir)
    results = []
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return results

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            # LLM 逻辑：找 gguf，看有没有 mmproj
            if sub_dir == "LLM":
                main_file = next((f for f in files if f.endswith('.gguf') and "mmproj" not in f.lower()), None)
                if main_file:
                    has_mm = any("mmproj" in f.lower() for f in files)
                    results.append({"id": folder, "has_vision": has_mm})
            # GEN 逻辑：找关键组件
            elif sub_dir == "GEN":
                if any(f.endswith('.gguf') or f.endswith('.safetensors') for f in files):
                    results.append({"id": folder})
    return results

# --- 在 main.py 的路由部分增加 ---


@app.get("/v1/models")
async def list_models():
    """
    一次性返回分类后的模型列表，供前端双下拉框使用
    """
    try:
        llm_list = scan_models("LLM")
        gen_list = scan_models("GEN")
        
        return {
            "llm": llm_list,
            "gen": gen_list,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ 扫描模型目录失败: {e}")
        return {"llm": [], "gen": [], "error": str(e)}


from fastapi import BackgroundTasks # 导入后台任务

def cleanup_temp_files(files):
    """物理删除函数"""
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 后台清理成功: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️ 后台清理失败: {e}")

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    messages = body.get("messages", [])

        # 1. 物理定位模型目录 (LM Studio 模式)
    # 前端现在传过来的是文件夹名，如 "Qwen3.6-27B-Vision"
    target_id = body.get("model")
    model_root = os.path.join(BASE_DIR, "models", "LLM", target_id)
    
    if not os.path.exists(model_root):
        return JSONResponse({"status": "error", "message": f"Engine path '{target_id}' not found."}, status_code=404)

    
    # 2. 扫描目录组件
    files = os.listdir(model_root)
    llm_path = next((os.path.join(model_root, f) for f in files if f.endswith('.gguf') and "mmproj" not in f.lower()), None)
    mmproj_path = next((os.path.join(model_root, f) for f in files if "mmproj" in f.lower() or "adapter" in f.lower()), None)


    
    # --- 1. 恢复并增强 full_text 提取逻辑 ---
    text_parts = []
    has_image = False
    # 用于记录本次请求涉及到的临时文件路径
    session_files = []

    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            # 处理多模态列表格式
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    has_image = True

                if "local_path" in item: 
                    session_files.append(item["local_path"])

    
    full_text = "".join(text_parts) # 变量回归，用于动态计算

    # ================= 🛡️ 安全防御核心逻辑 =================
    if has_image and not mmproj_path:
        print(f"🚨 拦截非法视觉请求: {target_id} 不具备多模态能力")
        
        async def security_rejection():
            error_msg = f"❌ **访问拒绝**：当前选中的模型 `{target_id}` 是纯文本引擎，不具备视觉分析能力。请在侧边栏切换至带 👁️ 图标的 Vision 模型后再试。"
            # 兼容 OpenAI 格式返回错误，防止调用方解析报错
            yield f"data: {json.dumps({'choices': [{'delta': {'content': error_msg}}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        
        return StreamingResponse(security_rejection(), media_type="text/event-stream")
    # =====================================================


    # 启动审计
    selected_model = body.get("model", os.path.basename(profile['llm']['path']))
    task_id = audit_log.start_task(selected_model, request.client.host, "VISION" if has_image else "TEXT")
    
    
    # --- 2. 动态参数计算 (找回 0.3 的灵魂) ---
    estimated_tokens = len(full_text) // 1.2
    # 视觉模式下，图片通常占 1000-2000 tokens，我们额外预留空间
    buffer_tokens = 4096 if has_image else 1024
    n_ctx = int(min(max(estimated_tokens + buffer_tokens, 4096), profile['llm']['n_ctx_limit']))
    
    # 动态 Batch
    n_batch = 4096 if n_ctx > 16384 else 1024

    # 获取选中的模型名
    selected_model = body.get("model")
    model_name = selected_model if selected_model else os.path.basename(profile['llm']['path'])
    
    # --- 核心逻辑：自动协议分发 ---
    handler_name, chat_fmt = get_vision_specs(model_name)
    adapter_path = os.path.abspath(profile['llm'].get('adapter_path', ''))
    
    handler = None
    if has_image and os.path.exists(adapter_path):
        # 动态实例化列表中的 Handler
        HandlerClass = getattr(lcf, handler_name, None)
        if HandlerClass:
            handler = HandlerClass(clip_model_path=adapter_path, verbose=False)
            print(f"🧬 自动适配协议: {handler_name} | Format: {chat_fmt}")
        else:
            # 如果当前库版本不支持该 Handler，回退到 Llava15 通用版
            handler = lcf.Llava15ChatHandler(clip_model_path=adapter_path)
            print(f"⚠️ 库版本不支持 {handler_name}，回退到通用版")


    
    # 分布式探测
    rpc_nodes_cfg = config['server'].get('rpc_nodes', [])
    rpc_param, tensor_split = get_balanced_rpc_config(rpc_nodes_cfg)

    import base64

    def path_to_base64(file_path):
        """将本地磁盘路径的图片转换为 Base64 Data URI"""
        ext = file_path.split('.')[-1].lower()
        mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
        try:
            with open(file_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"❌ 还原 Base64 失败: {e}")
            return None

    # --- 3. 定义流式生成器 (确保捕获外部变量) ---
    async def generate(n_ctx, n_batch, model_path, handler, chat_fmt, task_id): # 通过参数显式传递最稳
               # 增加信号保护，防止 CTRL+C 时产生孤儿进程
        llm = None
        try:
            # 1. 关键修复：在推理前，遍历 messages，将 local_path 替换回 Base64
            processed_messages = []
            for m in messages:
                new_msg = m.copy()
                if isinstance(m.get("content"), list):
                    new_content = []
                    for item in m["content"]:
                        if item.get("type") == "image_url":
                            # 检查是否有之前存入的 local_path
                            local_path = item["image_url"].get("url")
                            print(f"open img {local_path}")
                            # 如果 url 是一个本地路径（不以 http 或 data: 开头）
                            if local_path and not local_path.startswith(('http', 'data:')):
                                b64_uri = path_to_base64(local_path)
                                if b64_uri:
                                    new_content.append({"type": "image_url", "image_url": {"url": b64_uri}})
                                    continue
                        new_content.append(item)
                    new_msg["content"] = new_content
                processed_messages.append(new_msg)

            async with lock:
                start_time = time.time()
                
                # --- 步骤 1: 立即发送加载信号（此时 Llama 还没开始初始化） ---
                yield f"data: {json.dumps({'extra': {'status': 'LOADING_MODEL', 'active_model': os.path.basename(model_path)}})}\n\n"
                # 给网络一个小缓冲，确保前端渲染出 thoughtDiv
                await asyncio.sleep(0.1) 


                # --- 核心清理步骤 ---
                gc.collect()
                torch.mps.empty_cache() # 强制排空 Metal 显存残留

                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    rpc=rpc_param,
                    #tensor_split=tensor_split,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    flash_attn=True,
                    # 关键：加载视觉适配器
                    chat_handler=handler,
                    chat_format=chat_fmt,
                    offload_kqv=True,  # 确保视觉特征全量进入显存
                    verbose=False
                )

                load_done = time.time()
                load_ms = round((load_done - start_time) * 1000, 2)

                # --- 步骤 3: 立即发送预处理信号 ---
                yield f"data: {json.dumps({'extra': {
                    'status': 'PROCESSING_PROMPT', 
                    'processing_ms': load_ms, 
                    'n_ctx': n_ctx
                }})}\n\n"

                token_count = 0
                start_gen = time.time()
                    
                # --- 步骤 4: 推理循环 ---
                for chunk in llm.create_chat_completion(
                    messages=processed_messages, 
                    stream=True,
                    temperature=0.1 if has_image else 0.7, 
                    repeat_penalty=1.1, # 关键：禁用缓存重用，强制从零计算 Prompt
                    ):

                    if token_count == 0:
                        # 收到第一个 token，清理思考文字
                        yield f"data: {json.dumps({'extra': {'status': 'GENERATING'}})}\n\n"
                    
                    token_count += 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                
                # --- 步骤 5: 发送最终统计 ---
                gen_duration = time.time() - start_gen
                tps = round(token_count / gen_duration, 2) if gen_duration > 0 else 0
                audit_log.end_task(task_id, tps=tps, tokens=token_count)

                yield f"data: {json.dumps({'extra': {
                    'status': 'DONE',
                    'tps': tps, 
                    'total_tokens': token_count,
                    'gen_time': round(gen_duration, 2)
                }})}\n\n"

                yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"❌ 推理异常: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # 无论成功还是被 CTRL+C 中断，必须释放资源
            if llm:
                del llm
            gc.collect()
            torch.mps.empty_cache()
            print(f"♻️ 显存已释放 (Task: {task_id})")


    # 启动时传入计算好的参数
    # 启动生成
    selected_model = body.get("model")
    model_path = os.path.join(BASE_DIR, "models", selected_model) if selected_model else os.path.abspath(profile['llm']['path'])

    # 注册后台任务：告诉 FastAPI 在响应完全发送（或关闭）后运行此函数
    if session_files:
        background_tasks.add_task(cleanup_temp_files, session_files)

    return StreamingResponse(generate(n_ctx, n_batch, model_path, handler, chat_fmt, task_id), media_type="text/event-stream")


@app.post("/v1/images/generations")
async def image_gen_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    target_gen_id = body.get("model") # 从前端传来的 GEN 目录下文件夹名
    client_ip = request.client.host

    # 1. 物理路径探测
    gen_root = os.path.join(BASE_DIR, "models", "GEN", target_gen_id)
    if not os.path.exists(gen_root):
        return JSONResponse({"error": "Image engine not found"}, status_code=404)

    # 2. 启动监控审计
    task_id = audit_log.start_task(target_gen_id, client_ip, "IMAGE_GEN")

    # 3. 显存接力与生成逻辑
    async with lock:
        try:
            # 彻底清理 LLM 留下的显存
            gc.collect(); torch.mps.empty_cache()
            
            start_time = time.time()
            
            # 自动锁定组件 (适应不同命名习惯)
            files = os.listdir(gen_root)
            model_path = next(os.path.join(gen_root, f) for f in files if f.endswith('.gguf') or f.endswith('.safetensors'))
            
            # 初始化 SD 引擎 (基于 stable-diffusion-cpp)
            from stable_diffusion_cpp import StableDiffusion
            sd = StableDiffusion(
                diffusion_model_path="./models/GEN/z_image_turbo/z_image_turbo-Q8_0.gguf",
                llm_path="./models/GEN/z_image_turbo/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                vae_path="./models/GEN/z_image_turbo/ae.safetensors",
                offload_params_to_cpu=True,
                diffusion_flash_attn=True,
            )
            
            images = sd.generate_image(
                prompt=prompt,
                height=1024,
                width=512,
                cfg_scale=1.0, # a cfg_scale of 5 is recommended for Z-Image base (non-turbo)
            )
            
            # 保存到 static/generated 用于前端访问
            os.makedirs("static/generated", exist_ok=True)
            img_name = f"gen_{uuid.uuid4().hex}.png"
            img_path = f"static/generated/{img_name}"
            images[0].save(img_path)
            
            duration = time.time() - start_time
            print(f"🎨 生图完成: {duration:.2f}s | Path: {img_path}")
            
            # 结束审计
            audit_log.end_task(task_id, tps=0, tokens=1) # 生图任务不计 token
            
            return {"data": [{"url": f"/static/generated/{img_name}"}]}
            
        except Exception as e:
            print(f"❌ 生图异常: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            # 立即释放，为下一次对话腾出空间
            if 'sd' in locals(): del sd
            gc.collect(); torch.mps.empty_cache()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
        return templates.TemplateResponse(
        name="index.html", 
        context={"request": request, "profile": active_name}, request=request)

if __name__ == "__main__":
    upload_dir = os.path.join(BASE_DIR, "uploads")
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            try: os.remove(os.path.join(upload_dir, f))
            except: pass
    print("🧹 临时上传目录已初始化")

    uvicorn.run(app, host="0.0.0.0", port=config['server']['port'])

