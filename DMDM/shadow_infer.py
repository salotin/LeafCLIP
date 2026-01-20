# shadow_infer.py
import os
from os.path import join, basename, splitext
import argparse
import copy
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from kornia.morphology import dilation
from accelerate import Accelerator

from guided_diffusion.script_util import create_gaussian_diffusion
from diffusionSDRM import DensePosteriorConditionalUNet  # 


def bgr2rgb_float(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def hwc_to_chw_tensor(img_rgb01: np.ndarray) -> torch.Tensor:
    # HWC [0,1] -> CHW float32 tensor
    return torch.from_numpy(img_rgb01.transpose(2, 0, 1)).float()


@torch.no_grad()
def infer_one(
    ema_model,
    ema_feature_encoder,
    diffusion,
    device,
    img_path: str,
    mask_path: str | None,
    out_dir: str,
    ddim_steps: int,
    resize: int = 256,
):
    # ---- load shadow image (lq) ----
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_bgr = cv2.resize(img_bgr, (resize, resize), interpolation=cv2.INTER_CUBIC)
    img_rgb01 = bgr2rgb_float(img_bgr)
    lq = hwc_to_chw_tensor(img_rgb01).unsqueeze(0).to(device)  # [1,3,H,W]

    # normalize like dataset: mean/std=[0.5,0.5,0.5] :contentReference[oaicite:6]{index=6}
    normalize(lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

    # ---- load mask ----
    if mask_path is not None and os.path.exists(mask_path):
        m_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if m_bgr is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        m_bgr = cv2.resize(m_bgr, (resize, resize), interpolation=cv2.INTER_NEAREST)
        m_rgb01 = bgr2rgb_float(m_bgr)
        mask = hwc_to_chw_tensor(m_rgb01).unsqueeze(0).to(device)  # [1,3,H,W]（与训练脚本的 in_channels=3+3 保持一致可能性更大）
    else:
        # 若没有 mask，就用全0 mask（不建议，但能跑通）
        mask = torch.zeros((1, 3, resize, resize), device=device, dtype=torch.float32)

    # dilation like test :contentReference[oaicite:7]{index=7}
    mask = dilation(mask, torch.ones(21, 21, device=device))

    # ---- feature encoder ----
    t0 = torch.tensor([0], device=device, dtype=torch.long)
    intrinsic_feature = ema_feature_encoder(lq, t0, latent=mask)   # [1,1,H,W]

    # ---- diffusion sampling ----
    latent = torch.cat((lq, intrinsic_feature), dim=1)             # [1,4,H,W]
    pred = diffusion.ddim_sample_loop(
        ema_model,
        lq.shape,  # output shape uses x's shape: [1,3,H,W]
        model_kwargs={"latent": latent},
        progress=False,
    )

    # ---- denorm & save ----
    pred = pred.clamp(-1, 1)
    pred_255 = (pred / 2 + 0.5) * 255.0  # [0,255]
    pred_255 = pred_255.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC RGB

    pred_255 = np.clip(pred_255, 0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_255, cv2.COLOR_RGB2BGR)

    os.makedirs(out_dir, exist_ok=True)
    name = splitext(basename(img_path))[0]
    out_path = join(out_dir, f"{name}_deshadow.png")
    cv2.imwrite(out_path, pred_bgr)
    return out_path


def build_models(device):
    # 与 test 脚本保持一致的模型结构参数 :contentReference[oaicite:8]{index=8}
    model = DensePosteriorConditionalUNet(
        in_channels=3 + 3 + 1,
        out_channels=6,
        model_channels=192,
        num_res_blocks=2,
        attention_resolutions=[8, 16, 32],
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_model = copy.deepcopy(model)

    feature_encoder = DensePosteriorConditionalUNet(
        in_channels=3 + 3,
        out_channels=1,
        model_channels=96,
        num_res_blocks=1,
        attention_resolutions=[8, 16],
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_feature_encoder = copy.deepcopy(feature_encoder)

    ema_model.to(device)
    ema_feature_encoder.to(device)
    return model, ema_model, feature_encoder, ema_feature_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="accelerate save_state path, e.g. experiments/state_244999.bin")
    parser.add_argument("--in_dir", type=str, required=True, help="input shadow images folder")
    parser.add_argument("--mask_dir", type=str, default=None, help="mask folder (optional but recommended)")
    parser.add_argument("--out_dir", type=str, required=True, help="output folder")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps (match test: 50)")
    parser.add_argument("--resize", type=int, default=256)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")  # test 脚本是 fp16 :contentReference[oaicite:9]{index=9}
    device = accelerator.device

    model, ema_model, feature_encoder, ema_feature_encoder = build_models(device)
    model, ema_model, feature_encoder, ema_feature_encoder = accelerator.prepare(
        model, ema_model, feature_encoder, ema_feature_encoder
    )

    # 采样扩散器：test 用 ddim50 :contentReference[oaicite:10]{index=10}
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="linear",
        use_kl=False,
        timestep_respacing=f"ddim{args.ddim_steps}",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=0.5,
        p2_k=1,
    )

    # 加载 accelerate state（包含 ema_model / ema_feature_encoder 等）
    accelerator.load_state(args.ckpt)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    img_list = [join(args.in_dir, f) for f in os.listdir(args.in_dir) if f.lower().endswith(exts)]
    img_list.sort()

    if accelerator.is_main_process:
        print(f"=> Found {len(img_list)} images in {args.in_dir}")

    for img_path in img_list:
        mask_path = None
        if args.mask_dir is not None:
            mask_path = join(args.mask_dir, basename(img_path))  # 与原数据组织一致 

        out_path = infer_one(
            ema_model=ema_model,
            ema_feature_encoder=ema_feature_encoder,
            diffusion=diffusion,
            device=device,
            img_path=img_path,
            mask_path=mask_path,
            out_dir=args.out_dir,
            ddim_steps=args.ddim_steps,
            resize=args.resize,
        )

        if accelerator.is_main_process:
            print(f"[OK] {basename(img_path)} -> {out_path}")


if __name__ == "__main__":
    main()
