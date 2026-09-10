"""
Beautification Model Service — pixel-space InstructPix2Pix (stage1-512).

Input: multipart/form-data
  image:                image file (JPEG/PNG bytes)
  prompt:               English beautification instruction
  model:                deepfrr | ffhqr  (currently ignored, single model loaded)
  steps:                DDIM inference steps (default 30)
  guidance_scale:       text guidance (default 3.0)
  image_guidance_scale: image guidance (default 1.5)
  seed:                 random seed (default 42)
Output: raw PNG image bytes
"""

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response


_DEFAULT_PROMPT = "Please apply a natural beauty retouch with subtle skin smoothing and brighter skin."


# ---------------------------------------------------------------------------
# 模型（启动时加载一次）
# ---------------------------------------------------------------------------
_state = {}


def load_model():
    base = Path(os.getenv("BEAUTIFY_BASE_MODEL", "/home/xingfc/hf_models/timbrooks_instruct-pix2pix_fp16"))
    unet_dir = Path(os.getenv(
        "BEAUTIFY_UNET_DIR",
        "/home/xingfc/beauty_models/beauty-pixel-ip2p-stage1-512-ffhqr-other-fromscratch-h200/checkpoint-11700/unet",
    ))
    scheduler_dir = os.getenv("BEAUTIFY_SCHEDULER_DIR") or str(unet_dir.parent / "scheduler")

    device = torch.device(os.getenv("BEAUTIFY_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[beautify] loading on {device} ...", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", variant="fp16", torch_dtype=dtype, local_files_only=True
    ).to(device)
    text_encoder.eval()
    unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype, local_files_only=True).to(device)
    unet.eval()
    scheduler = DDIMScheduler.from_pretrained(scheduler_dir, local_files_only=True)
    scheduler.config.clip_sample = False
    print("[beautify] model loaded.", flush=True)

    _state.update(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        resolution=int(os.getenv("BEAUTIFY_RESOLUTION", "512")),
    )


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def match_color(out, ref):
    """按通道把 out 的颜色统计(均值/标准差)对齐到 ref，去掉整体偏色。"""
    out = out.astype("float32")
    ref = ref.astype("float32")
    for c in range(3):
        om = out[..., c].mean()
        os_ = out[..., c].std() + 1e-6
        rm = ref[..., c].mean()
        rs = ref[..., c].std() + 1e-6
        out[..., c] = (out[..., c] - om) / os_ * rs + rm
    return out.clip(0, 255).round().astype("uint8")

def run_beautify(image_pil: Image.Image, prompt: str, num_steps: int, guidance: float, image_guidance: float, seed: int) -> Image.Image:
    tok = _state["tokenizer"]
    text_encoder = _state["text_encoder"]
    unet = _state["unet"]
    scheduler = _state["scheduler"]
    device = _state["device"]
    dtype = _state["dtype"]
    resolution = _state["resolution"]

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    original = tfm(image_pil.convert("RGB"))
    cond = original.unsqueeze(0).to(device=device, dtype=dtype)
    empty = torch.zeros_like(cond)

    def tokenize(text):
        return tok(
            text,
            max_length=tok.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

    with torch.no_grad():
        prompt_emb = text_encoder(tokenize(prompt))[0]
        empty_emb = text_encoder(tokenize(""))[0]

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(num_steps, device=device)
    image = torch.randn(cond.shape, generator=generator, device=device, dtype=dtype)

    for timestep in scheduler.timesteps:
        image_model_input = torch.cat([image, image, image], dim=0)
        cond_model_input = torch.cat([cond, cond, empty], dim=0)
        model_input = torch.cat([image_model_input, cond_model_input], dim=1)
        text_input = torch.cat([prompt_emb, empty_emb, empty_emb], dim=0)
        with torch.no_grad():
            noise_pred = unet(model_input, timestep.expand(3), text_input).sample
        noise_text, noise_image, noise_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_uncond
            + guidance * (noise_text - noise_image)
            + image_guidance * (noise_image - noise_uncond)
        )
        image = scheduler.step(noise_pred, timestep, image).prev_sample

    tensor = image[0].detach().float().clamp(-1, 1)
    tensor = (tensor + 1) / 2
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")

    # 颜色校正：把输出颜色对齐到输入，去掉黄/绿偏色
    ref = original.detach().float().clamp(-1, 1)
    ref = (ref + 1) / 2
    ref_array = (ref.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    array = match_color(array, ref_array)

    return Image.fromarray(array)


# ---------------------------------------------------------------------------
# 启动时加载模型
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Beautification Model Service", version="0.3.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.post("/beautify")
async def beautify(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("ffhqr"),
    steps: int = Form(30),
    guidance_scale: float = Form(3.0),
    image_guidance_scale: float = Form(1.5),
    seed: int = Form(42),
):
    img_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    prompt_text = prompt.strip() or _DEFAULT_PROMPT
    print(f"[beautify] model={model} steps={steps} guidance={guidance_scale} img_guidance={image_guidance_scale} seed={seed}", flush=True)

    t0 = time.time()
    result = run_beautify(image_pil, prompt_text, steps, guidance_scale, image_guidance_scale, seed)
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[beautify] done in {elapsed_ms:.0f} ms", flush=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(_state)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
