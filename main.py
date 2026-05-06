import gc, os, time, torch, asyncio, base64, yaml, json, psutil, socket
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
import uvicorn
from fastapi.staticfiles import StaticFiles

app = FastAPI()
lock = asyncio.Lock()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory="static"), name="static")

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


# 修改 main.py 中的 chat_endpoint 生成器部分
async def generate():
    async with lock:
        start_time = time.time()
        # 1. 模拟处理进度：发送正在加载信号
        yield f"data: {json.dumps({'extra': {'status': 'LOADING_MODEL', 'msg': '正在初始化架构...'}})}\n\n"
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, rpc=rpc_param, tensor_split=tensor_split,
            n_ctx=n_ctx, n_batch=n_batch, flash_attn=True, verbose=False
        )
        load_done = time.time()
        
        # 2. 模拟 PP (Prompt Processing) 进度包
        # 提示：对于长文本，此阶段最耗时。我们发送一个预估包。
        yield f"data: {json.dumps({'extra': {
            'status': 'PROCESSING_PROMPT',
            'msg': f'正在处理长文本 (CTX: {n_ctx})...',
            'processing_ms': round((load_done - start_time) * 1000, 2)
        }})}\n\n"

        token_count = 0
        start_gen = time.time()
        
        # 开始推理
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            if token_count == 0:
                # 收到第一个 token 时，告诉前端清理“思考/处理”文字
                yield f"data: {json.dumps({'extra': {'status': 'GENERATING'}})}\n\n"
            
            token_count += 1
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # 3. 最终统计包
        duration = time.time() - start_gen
        tps = round(token_count / duration, 2) if duration > 0 else 0
        yield f"data: {json.dumps({'extra': {
            'status': 'DONE',
            'tps': tps, 
            'total_tokens': token_count,
            'gen_time': round(duration, 2)
        }})}\n\n"
        yield "data: [DONE]\n\n"
        
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
    
    # --- 核心改进：获取前端选中的模型 ---
    # 优先级：请求体指定 > Profile默认值
    selected_model = body.get("model")
    if selected_model and selected_model.endswith(".gguf"):
        model_path = os.path.join(BASE_DIR, "models", selected_model)
    else:
        model_path = os.path.abspath(profile['llm']['path'])

    # 动态参数计算
    full_text = "".join([m.get("content", "") for m in messages])
    n_ctx = int(min(max(len(full_text)//1.2 + 1024, 4096), profile['llm']['n_ctx_limit']))
    
    # 分布式探测
    rpc_nodes_cfg = config['server'].get('rpc_nodes', [])
    rpc_param, tensor_split = get_balanced_rpc_config(rpc_nodes_cfg)

    async def generate():
        async with lock:
            start_load = time.time()
            llm = Llama(
                model_path=model_path, # 使用选中的模型
                n_gpu_layers=-1,
                rpc=rpc_param,
                tensor_split=tensor_split,
                n_ctx=n_ctx,
                n_batch=profile['llm']['n_batch'],
                flash_attn=True,
                verbose=False
            )
            load_ms = round((time.time() - start_load) * 1000, 2)
            
            # 向前端传回当前正在运行的模型文件名
            yield f"data: {json.dumps({'extra': {'active_model': os.path.basename(model_path), 'processing_ms': load_ms, 'n_ctx': n_ctx}})}\n\n"

            token_count = 0
            start_gen = time.time()
            for chunk in llm.create_chat_completion(messages=messages, stream=True):
                token_count += 1
                yield f"data: {json.dumps(chunk)}\n\n"
            
            tps = round(token_count / (time.time() - start_gen), 2)
            yield f"data: {json.dumps({'extra': {'tps': tps}})}\n\n"
            yield "data: [DONE]\n\n"
            
            del llm
            gc.collect()
            torch.mps.empty_cache()

    return StreamingResponse(generate(), media_type="text/event-stream")

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

