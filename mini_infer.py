#!/usr/bin/env python3
"""最小推理验证脚本 — 绕开 infer_unified.py，直接用 app_inference 路径。"""

import os, sys, warnings
warnings.filterwarnings("ignore")

os.environ.update({
    "APP_ENABLED":           "1",
    "APP_MODEL_NAME":        "LHMPP-700M",
    "APP_MODEL_CONFIG":      "./configs/train/LHMPP-any-view.yaml",
    "APP_TYPE":              "infer.human_lrm_a4o",
    "NUMBA_THREADING_LAYER": "omp",
    "MODEL_PATH":            "/temp/hanmo/models/lhm_plusplus/LHMPP-700M",
    "TORCH_HOME":            "/cache/hanmo/torch",
})

sys.path.insert(0, "./")
import torch
torch._dynamo.config.disable = True   # 与原始脚本保持一致
import numpy as np
import imageio.v3 as iio
from PIL import Image

# ── 1. 配置 ───────────────────────────────────────────────────────────────────
print("[1/5] 解析配置 ...")
from omegaconf import OmegaConf

cfg = OmegaConf.create()
cfg_train = OmegaConf.load("./configs/train/LHMPP-any-view.yaml")
cfg.source_size   = cfg_train.dataset.source_image_res  # 512
cfg.render_size   = cfg_train.dataset.render_image.high  # 420
cfg.src_head_size = getattr(cfg_train.dataset, "src_head_size", 112)
cfg.model_name    = "/temp/hanmo/models/lhm_plusplus/LHMPP-700M"
cfg.motion_video_read_fps = 30
cfg.setdefault("logger", "INFO")
print(f"  source_size={cfg.source_size}, render_size={cfg.render_size}")

# ── 2. 加载模型 ───────────────────────────────────────────────────────────────
print("[2/5] 加载模型 ...")
from accelerate import PartialState
PartialState()  # accelerate logging 依赖此初始化
from core.utils.hf_hub import wrap_model_hub
from core.models import model_dict

model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
lhm = model_cls.from_pretrained(cfg.model_name)
lhm.cuda().eval()
print(f"  VRAM used ≈ {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── 3. 参考图像 ───────────────────────────────────────────────────────────────
print("[3/5] 加载参考图像 ...")
img = np.array(
    Image.open("assets/example_aigc_images/055330c0d988_000.jpg")
         .convert("RGB")
         .resize((cfg.source_size, cfg.source_size))
)
ref_imgs_tensor = (
    torch.from_numpy(img / 255.0).permute(2, 0, 1).float().unsqueeze(0)
)  # (1, 3, H, W)
print(f"  shape={tuple(ref_imgs_tensor.shape)}")

# ── 4. Motion 序列 ────────────────────────────────────────────────────────────
print("[4/5] 加载 motion (Dance_I, 30 frames) ...")
from core.utils.app_utils import get_motion_information

MOTION_SIZE  = 30
motion_mp4   = "./motion_video/Dance_I/samurai_visualize.mp4"
motion_name, motion_seqs = get_motion_information(
    motion_mp4, cfg, motion_size=MOTION_SIZE
)

smplx_params = motion_seqs["smplx_params"]
# betas shape 可能是 (10,) 或 (T, 10) — 取第一帧
betas = smplx_params["betas"]
if betas.dim() == 2:
    betas = betas[0]
smplx_params["betas"] = betas.unsqueeze(0).float().cuda()

video_size = min(MOTION_SIZE, len(motion_seqs["motion_seqs"]))
print(f"  motion='{motion_name}', rendering={video_size} frames")

# ── 5. 推理 ───────────────────────────────────────────────────────────────────
print("[5/5] 推理中 ...")
from scripts.inference.app_inference import inference_results

with torch.no_grad():
    rgbs = inference_results(
        lhm,
        ref_imgs_tensor,
        smplx_params,
        motion_seqs,
        video_size=video_size,
        device="cuda",
    )

# ── 保存 ──────────────────────────────────────────────────────────────────────
os.makedirs("./debug", exist_ok=True)
out = "./debug/mini_infer_out.mp4"
iio.imwrite(out, rgbs, fps=15, codec="libx264", pixelformat="yuv420p")
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n[OK] frames={rgbs.shape}  Peak VRAM={peak:.1f} GB  => {out}")
