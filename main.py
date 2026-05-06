import gc, os, time, torch, asyncio, base64, yaml, json, psutil, socket
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
import uvicorn
from fastapi.staticfiles import StaticFiles
from llama_cpp.llama_chat_format import Llava15ChatHandler

app = FastAPI()
lock = asyncio.Lock()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory="static"), name="static")

import signal

def graceful_exit(sig, frame):
    print("\n正在关闭 MetalHub，清理显存...")
    # 这里可以尝试强制释放一些全局资源
    gc.collect()
    torch.mps.empty_cache()
    os._exit(0)

# 在 main 块中注册信号
signal.signal(signal.SIGINT, graceful_exit) # 捕捉 Ctrl+C


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

# ================= 监控实时流接口 =================
@app.get("/v1/system/monitor/stream")
async def monitor_stream():
    """专门供前端控制台连接的 SSE 接口"""
    async def event_generator():
        # 首次连接，发送当前所有任务
        yield f"data: {json.dumps({'type': 'INIT', 'tasks': list(audit_log.active_tasks.values()), 'total': audit_log.total_tokens_all_time})}\n\n"
        
        while True:
            # 阻塞等待队列中的新更新
            update = await audit_log.broadcast_queue.get()
            yield f"data: {json.dumps({'type': 'UPDATE', 'task': update, 'total': audit_log.total_tokens_all_time})}\n\n"
            
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

# --- 在 main.py 的路由部分增加 ---

@app.get("/v1/models")
async def list_models():
    """扫描 models 目录并返回 OpenAI 兼容的模型列表"""
    model_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(model_dir):
        return {"data": []}
    
    files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    return {
        "object": "list",
        "data": [{"id": f, "object": "model", "created": int(time.time()), "owned_by": "metalhub"} for f in files]
    }

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # --- 1. 恢复并增强 full_text 提取逻辑 ---
    text_parts = []
    has_image = False
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
    
    full_text = "".join(text_parts) # 变量回归，用于动态计算


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

    # --- 3. 定义流式生成器 (确保捕获外部变量) ---
    async def generate(n_ctx, n_batch, model_path, handler, chat_fmt, task_id): # 通过参数显式传递最稳
               # 增加信号保护，防止 CTRL+C 时产生孤儿进程
        llm = None
        try:
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
                    messages=messages, 
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
    return StreamingResponse(generate(n_ctx, n_batch, model_path, handler, chat_fmt, task_id), media_type="text/event-stream")

# ================= 3. 生图接口 (保持原有逻辑) =================
@app.post("/v1/images/generations")
async def image_endpoint(request: Request):
    async with lock:
        body = await request.json()
        cfg = profile['image']
        sd = StableDiffusion(
            gguf_model_path=os.path.abspath(cfg['gguf_path']),
            t5xxl_path=os.path.abspath(cfg['t5_path']),
            vae_path=os.path.abspath(cfg['vae_path']),
            flash_attn=True, n_threads=8
        )
        try:
            image = sd.txt2img(prompt=body.get("prompt",""), steps=cfg['steps'], cfg_scale=cfg['cfg_scale'])
            buf = BytesIO(); image.save(buf, format="PNG")
            return {"data": [{"url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}]}
        finally:
            del sd
            gc.collect()
            torch.mps.empty_cache()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
        return templates.TemplateResponse(
        name="index.html", 
        context={"request": request, "profile": active_name}, request=request)

if __name__ == "__main__":
    print(f"DEBUG: Memory detected: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    uvicorn.run(app, host="0.0.0.0", port=config['server']['port'])

