import gc, os, time, torch, asyncio, base64, yaml, json, psutil
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
import uvicorn

app = FastAPI()
lock = asyncio.Lock()

# 获取当前脚本所在目录，确保路径在不同环境下都能正确识别
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ================= 1. 硬件探测与配置加载 =================
def detect_hardware():
    total_mem = psutil.virtual_memory().total / (1024**3)
    if total_mem >= 48: return "high_end"
    elif total_mem >= 24: return "mid_range"
    else: return "low_end"

# 加载 config.yaml
config_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

active_profile_name = config['server'].get('active_profile', 'auto')
if active_profile_name == 'auto':
    active_profile_name = detect_hardware()

profile = config['profiles'][active_profile_name]

# ================= 2. 文本对话接口 (支持流式) =================
@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # 动态上下文计算
    full_text = "".join([m.get("content", "") for m in messages])
    tokens = len(full_text) // 1.2
    n_ctx = int(min(max(tokens + 1024, 4096), profile['llm']['n_ctx_limit']))
    
    async def generate():
        async with lock:
            start_load = time.time()
            llm = Llama(
                model_path=os.path.abspath(profile['llm']['path']),
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                n_batch=profile['llm']['n_batch'],
                flash_attn=True,
                verbose=False
            )
            load_ms = round((time.time() - start_load) * 1000, 2)
            yield f"data: {json.dumps({'extra': {'processing_ms': load_ms, 'n_ctx': n_ctx}})}\n\n"

            token_count = 0
            start_gen = time.time()
            # 兼容 OpenAI 流式格式
            for chunk in llm.create_chat_completion(messages=messages, stream=True):
                token_count += 1
                yield f"data: {json.dumps(chunk)}\n\n"
            
            gen_duration = time.time() - start_gen
            tps = round(token_count / gen_duration, 2) if gen_duration > 0 else 0
            yield f"data: {json.dumps({'extra': {'tps': tps}})}\n\n"
            yield "data: [DONE]\n\n"
            
            del llm
            gc.collect()
            torch.mps.empty_cache()

    return StreamingResponse(generate(), media_type="text/event-stream")

# ================= 3. 生图接口 (显存接力) =================
@app.post("/v1/images/generations")
async def image_endpoint(request: Request):
    async with lock:
        body = await request.json()
        img_cfg = profile['image']
        
        sd = StableDiffusion(
            gguf_model_path=os.path.abspath(img_cfg['gguf_path']),
            t5xxl_path=os.path.abspath(img_cfg['t5_path']),
            vae_path=os.path.abspath(img_cfg['vae_path']),
            flash_attn=True, n_threads=8
        )
        try:
            # 确保 prompt 处理
            prompt = body.get("prompt", "")
            image = sd.txt2img(prompt=prompt, steps=img_cfg['steps'], cfg_scale=img_cfg['cfg_scale'])
            buf = BytesIO()
            image.save(buf, format="PNG")
            return {"data": [{"url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}]}
        finally:
            del sd
            gc.collect()
            torch.mps.empty_cache()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "profile": active_profile_name})

# ================= 4. MAIN 入口 =================
if __name__ == "__main__":
    port = config['server'].get('port', 11434)
    print(f"\n" + "="*50)
    print(f" MetalHub Engine 正在启动...")
    print(f" 运行地址: http://0.0.0:{port}")
    print(f" 硬件预设: {active_profile_name}")
    print(f" 使用uv运行: uv run python main.py")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

