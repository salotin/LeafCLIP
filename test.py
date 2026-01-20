import os
import numpy as np
import open_clip
import torch
import yaml
from easydict import EasyDict
from models.Necker import Necker
from models.Adapter import Adapter
import math
import argparse
import warnings
from utils.misc_helper import *
from datasets.dataset import PlantTestDataset
from torch.utils.data import DataLoader
from models.MapMaker import MapMaker
import pprint
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.metrics import average_precision_score, f1_score
import time  # 用于 FPS 计算

warnings.filterwarnings('ignore')


def normalization(segmentations, image_size, avgpool_size=128):
    """
    将 anomaly map 标准化并下采样 + 高斯平滑，用于可视化
    """
    segmentations = torch.tensor(segmentations[:, None, ...]).cuda()  # N x 1 x H x W
    segmentations = F.interpolate(
        segmentations,
        (image_size, image_size),
        mode='bilinear',
        align_corners=True
    )
    segmentations_ = F.avg_pool2d(
        segmentations,
        (avgpool_size, avgpool_size),
        stride=1
    ).cpu().numpy()

    min_scores = segmentations_.reshape(-1).min(axis=-1).reshape(1)
    max_scores = segmentations_.reshape(-1).max(axis=-1).reshape(1)

    segmentations = segmentations.squeeze(1).cpu().numpy()
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations = np.clip(segmentations, a_min=0, a_max=1)
    return cv2.GaussianBlur(segmentations, (5, 5), 0)


def evaluate_metrics(y_true, y_pred, threshold=0.5):
    """
    计算 mAP 和 F1 分数（宏平均）。
    y_true: list of 0/1
    y_pred: list of floats
    """
    # mAP 用连续分数
    mAP = average_precision_score(y_true, y_pred)
    # 二值化后计算 F1
    y_pred_bin = [1 if p >= threshold else 0 for p in y_pred]
    f1 = f1_score(y_true, y_pred_bin, average='macro')
    return mAP, f1


@torch.no_grad()
def make_vision_takens_info(model, model_cfg, layers_out):
    img = torch.ones(
        (1, 3, model_cfg['vision_cfg']['image_size'], model_cfg['vision_cfg']['image_size'])
    ).to(model.device)
    _, tokens = model.encode_image(img, layers_out)
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg['embed_dim']
    print(f"model token size is {model.token_size}  model token dim is {model.token_c}")


@torch.no_grad()
def validate(args, dataset_name, test_dataloader,
             clip_model, necker, adapter, prompt_maker, map_maker):
    """
    前向推理并收集分数、GT，返回包含 AUROC、pixel AUROC、mAP、F1 的字典
    """
    image_preds, image_gts = [], []
    pixel_preds, pixel_gts = [], []
    image_paths = []

    for batch in tqdm(test_dataloader, desc=f"Forward {dataset_name}"):
        images = batch['images'].to(clip_model.device)
        image_paths += batch['image_path']

        # 前向
        _, image_tokens = clip_model.encode_image(images, out_layers=args.config.layers_out)
        feats = necker(image_tokens)
        vis_feats = adapter(feats)
        prm_feats = prompt_maker(vis_feats)
        anomaly_map = map_maker(vis_feats, prm_feats)

        B, _, H, W = anomaly_map.shape
        # 只保留异常通道
        anomaly_map = anomaly_map[:, 1, :, :]
        pixel_preds.append(anomaly_map)

        scores, _ = torch.max(anomaly_map.view(B, H * W), dim=-1)
        image_preds += scores.cpu().tolist()
        image_gts += batch['is_anomaly'].cpu().tolist()

        if dataset_name == 'busi':
            pixel_gts.append(batch['mask'].cpu().numpy())

    # 准备 pixel-level 数据
    pixel_preds_np = [p.cpu().numpy() for p in pixel_preds]
    pixel_preds_norm = normalization(torch.cat(pixel_preds, 0), args.config.image_size)
    if dataset_name == 'busi':
        pixel_gts = np.concatenate(pixel_gts, axis=0)

    # 可视化保存
    save_root = os.path.join(args.vis_save_root, dataset_name)
    os.makedirs(save_root, exist_ok=True)
    for idx, (path, gt, pred_map) in enumerate(zip(image_paths, image_gts, pixel_preds_norm)):
        img = Image.open(path).convert("RGB").resize((args.config.image_size,)*2)
        img_np = np.array(img).astype(np.uint8)
        heat = show_cam_on_image(img_np / 255, pred_map, use_rgb=True)
        label = "normal" if gt == 0 else "abnormal"
        merged = [img_np, heat]
        if dataset_name == 'busi':
            merged.append(np.repeat(pixel_gts[idx][:, :, None], 3, axis=2) * 255)
        Image.fromarray(np.concatenate(merged, axis=1).astype(np.uint8)).save(
            os.path.join(save_root, f"{idx}_{label}_{os.path.basename(path)}")
        )

    # 计算原有指标：AUROC
    metric = compute_imagewise_metrics(image_preds, image_gts)
    if dataset_name == 'busi':
        metric.update(compute_pixelwise_metrics(pixel_preds_np, pixel_gts))

    # 计算 mAP 和 F1
    mAP, f1 = evaluate_metrics(image_gts, image_preds)
    metric['mAP'] = mAP
    metric['f1'] = f1

    return metric


@torch.no_grad()
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.config.random_seed)

    # 初始化 CLIP 模型
    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        args.config.model_name, args.config.image_size, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    args.config.model_cfg = model_cfg
    make_vision_takens_info(model, model_cfg, args.config.layers_out)

    # 初始化 necker / adapter / prompt_maker / map_maker
    necker = Necker(clip_model=model).to(device)
    adapter = Adapter(clip_model=model, target=model_cfg['embed_dim']).to(device)
    if args.config.prompt_maker != 'coop':
        raise NotImplementedError("currently only support 'coop'")
    from models.CoOp import PromptMaker
    prompt_maker = PromptMaker(
        prompts=args.config.prompts, clip_model=model,
        n_ctx=args.config.n_learnable_token, CSC=args.config.CSC,
        class_token_position=args.config.class_token_positions
    ).to(device)
    map_maker = MapMaker(image_size=args.config.image_size).to(device)

    # 加载 checkpoint
    ck = torch.load(args.checkpoint_path, map_location=device)
    adapter.load_state_dict(ck['adapter_state_dict'])
    prompt_maker.prompt_learner.load_state_dict(ck['prompt_state_dict'])
    adapter.eval(); prompt_maker.eval()

    # 针对每个数据集做测试
    for ds in args.config.test_datasets:
        if ds not in ['plant', 'busi']:
            raise NotImplementedError(f"Unsupported dataset {ds}")
        test_ds = PlantTestDataset(
            args=args.config,
            source=os.path.join(args.config.data_root, ds),
            preprocess=preprocess
        )
        loader = DataLoader(test_ds, batch_size=args.config.batch_size, num_workers=2)

        # FPS 计时
        t0 = time.time()
        res = validate(args, ds, loader, model, necker, adapter, prompt_maker, map_maker)
        t1 = time.time()
        fps = len(loader.dataset) / (t1 - t0)

        # 打印含所有指标
        if ds != 'plant':
            print(f"{ds}, images auroc: {res['images-auroc']:.4f}, "
                  f"mAP: {res['mAP']:.4f}, F1: {res['f1']:.4f}, FPS: {fps:.2f}")
        else:
            print(f"{ds}, images auroc: {res['images-auroc']:.4f}, "
                  f"pixel_auroc: {res['pixel-auroc']:.4f}, "
                  f"mAP: {res['mAP']:.4f}, F1: {res['f1']:.4f}, FPS: {fps:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test MediCLIP")
    parser.add_argument("--config_path",    type=str, required=True, help="model configs")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--vis_save_root",  type=str, default="vis_results", help="save visualization")
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
