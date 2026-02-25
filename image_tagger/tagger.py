from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json

import numpy as np
import pandas as pd
import timm
import torch
from safetensors.torch import load_file
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

# ======================== 核心配置区 ========================
# 1. 模型相关（相对路径，适配你的目录结构）
MODEL_DIR_REL = "./models--SmilingWolf--wd-eva02-large-tagger-v3"  # 模型文件夹相对路径

# 2. 图片与输出配置
IMAGE_DIR_REL = r"image_tagger\sample_image"  # 待处理图片文件夹
OUTPUT_DIR_REL = r"image_tagger\tags_output"  # JSON输出文件夹
GEN_THRESHOLD = 0.35  # 通用标签阈值
CHAR_THRESHOLD = 0.75  # 角色标签阈值
SUPPORTED_IMG_EXT = ["jpg", "jpeg", "png", "webp", "bmp"]  # 支持的图片格式
# ===========================================================================

# 设备配置（自动检测GPU/CPU）
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """确保图片为RGB格式，处理透明通道/调色板"""
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    """将图片填充为正方形（白色背景）"""
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    """标签数据结构"""
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]
    num_labels: int  # 新增：标签总数


def load_labels_local() -> LabelData:
    """从本地模型目录递归查找并加载selected_tags.csv"""
    model_dir = Path(MODEL_DIR_REL).resolve()
    csv_paths = list(model_dir.rglob("selected_tags.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"未找到标签文件！请检查 {model_dir} 及其子目录下是否有selected_tags.csv"
        )

    df = pd.read_csv(csv_paths[0], usecols=["name", "category"])
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
        num_labels=len(df)  # 获取实际标签数量
    )


def load_model_local(num_labels: int) -> nn.Module:
    """从本地加载eva02模型（适配实际标签维度，仅用本地文件，不下载）"""
    model_dir = Path(MODEL_DIR_REL).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 1. 查找配置文件和权重文件
    config_paths = list(model_dir.rglob("config.json"))
    weight_paths = {
        "safetensors": list(model_dir.rglob("*.safetensors")),
        "msgpack": list(model_dir.rglob("*.msgpack")),
        "bin": list(model_dir.rglob("*.bin")),
        "pth": list(model_dir.rglob("*.pth")),
    }

    # 校验必要文件
    if not config_paths:
        raise FileNotFoundError(f"在{model_dir}未找到config.json")
    if not any(weight_paths.values()):
        raise FileNotFoundError(f"在{model_dir}未找到模型权重文件（.safetensors/.msgpack/.bin/.pth）")

    # 2. 优先加载safetensors（你的目录里有该文件）
    if weight_paths["safetensors"]:
        weight_path = weight_paths["safetensors"][0]
        state_dict = load_file(weight_path)
    else:
        weight_path = weight_paths["msgpack"][0] if weight_paths["msgpack"] else weight_paths["bin"][0]
        state_dict = torch.load(weight_path, map_location="cpu")

    # 3. 创建适配实际标签数量的EVA02模型
    model = timm.create_model(
        "eva02_large_patch14_448",
        pretrained=False,
        num_classes=num_labels  # 关键修复：使用实际标签数量，而非固定14112
    ).eval()

    # 适配权重格式（处理可能的state_dict嵌套）
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 关键修复：过滤掉不匹配的层（仅保留能加载的权重）
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
        else:
            print(f"⚠️ 跳过不匹配的权重: {k} (checkpoint shape: {v.shape}, model shape: {model_state_dict.get(k, '不存在').shape})")

    # 加载过滤后的权重
    model.load_state_dict(filtered_state_dict, strict=False)

    # 移到指定设备
    model = model.to(torch_device)
    return model


def get_tags(probs: Tensor, labels: LabelData) -> Dict[str, Any]:
    """解析模型输出为标签结果，返回JSON可序列化的字典"""
    probs_np = probs.cpu().numpy()

    # 评分标签（safe/sensitive等）
    rating_labels = {labels.names[i]: float(probs_np[i]) for i in labels.rating if i < len(probs_np)}

    # 通用标签（阈值过滤+排序）
    gen_labels = {
        labels.names[i]: float(probs_np[i])
        for i in labels.general
        if i < len(probs_np) and probs_np[i] > GEN_THRESHOLD
    }
    gen_labels = dict(sorted(gen_labels.items(), key=lambda x: x[1], reverse=True))

    # 角色标签（阈值过滤+排序）
    char_labels = {
        labels.names[i]: float(probs_np[i])
        for i in labels.character
        if i < len(probs_np) and probs_np[i] > CHAR_THRESHOLD
    }
    char_labels = dict(sorted(char_labels.items(), key=lambda x: x[1], reverse=True))

    # 生成caption和格式化标签
    combined_names = list(gen_labels.keys()) + list(char_labels.keys())
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return {
        "caption": caption,
        "taglist": taglist,
        "ratings": rating_labels,
        "character_tags": char_labels,
        "general_tags": gen_labels,
        "gen_threshold": GEN_THRESHOLD,
        "char_threshold": CHAR_THRESHOLD
    }


def process_single_image(img_path: Path, model: nn.Module, transform, labels: LabelData):
    """处理单张图片，生成同名JSON标注文件"""
    try:
        # 加载并预处理图片
        img = Image.open(img_path)
        img = pil_ensure_rgb(img)
        img = pil_pad_square(img)
        inputs = transform(img).unsqueeze(0).to(torch_device)
        inputs = inputs[:, [2, 1, 0]]  # RGB转BGR

        # 模型推理
        with torch.inference_mode():
            outputs = model(inputs)
            outputs = F.sigmoid(outputs).squeeze(0)

        # 解析标签
        tag_result = get_tags(outputs, labels)

        # 生成输出路径（同名JSON）
        output_dir = Path(OUTPUT_DIR_REL).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{img_path.stem}.json"

        # 保存JSON文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tag_result, f, ensure_ascii=False, indent=4)

        return True, img_path.name, json_path.name

    except Exception as e:
        return False, img_path.name, str(e)


def main():
    """主函数：一键运行所有逻辑"""
    # 1. 校验目录
    image_dir = Path(IMAGE_DIR_REL).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    # 2. 先加载标签（获取实际标签数量），再加载模型
    print("🔍 加载本地标签文件...")
    labels = load_labels_local()
    print(f"📌 检测到标签数量: {labels.num_labels}")

    print("🔍 加载本地模型（适配标签维度）...")
    model = load_model_local(labels.num_labels)

    # 3. 创建图片预处理规则
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # 4. 精准查找所有图片（去重+过滤隐藏文件）
    image_files = []
    # 统一转为小写扩展名，避免重复
    ext_set = set([ext.lower() for ext in SUPPORTED_IMG_EXT])

    for file in image_dir.iterdir():
        # 跳过目录和隐藏文件
        if file.is_dir() or file.name.startswith("."):
            continue
        # 获取文件扩展名（小写）
        file_ext = file.suffix.lstrip(".").lower()
        if file_ext in ext_set:
            image_files.append(file)

    # 去重（避免大小写扩展名重复）
    image_files = list(dict.fromkeys(image_files))

    # 打印详细的图片列表
    print(f"\n📂 扫描到图片目录: {image_dir}")
    print(f"📋 支持的格式: {', '.join(SUPPORTED_IMG_EXT)}")
    print(f"🔢 检测到有效图片数量: {len(image_files)}")

    if len(image_files) > 0:
        print("📝 图片列表:")
        for idx, img_file in enumerate(image_files, 1):
            print(f"   {idx}. {img_file.name}")
    else:
        print(f"⚠️ 在 {image_dir} 中未找到任何支持的图片文件")
        return

    # 5. 批量处理图片（精准统计）
    print(f"\n🚀 开始处理 {len(image_files)} 张图片...")
    success_count = 0
    fail_list = []

    for idx, img_file in enumerate(image_files, 1):
        success, img_name, msg = process_single_image(img_file, model, transform, labels)
        if success:
            success_count += 1
            print(f"[{idx}/{len(image_files)}] ✅ 处理完成: {img_name} -> {msg}")
        else:
            fail_list.append((img_name, msg))
            print(f"[{idx}/{len(image_files)}] ❌ 处理失败: {img_name} - {msg}")

    # 6. 输出精准的统计结果
    print(f"\n📊 处理结果统计:")
    print(f"   总计扫描到: {len(image_files)} 张")
    print(f"   成功处理: {success_count} 张")
    print(f"   处理失败: {len(fail_list)} 张")

    if fail_list:
        print(f"\n❌ 失败详情:")
        for img_name, err_msg in fail_list:
            print(f"   {img_name}: {err_msg[:100]}...")  # 截断过长的错误信息

    print(f"\n📁 标注文件保存路径: {Path(OUTPUT_DIR_REL).resolve()}")


if __name__ == "__main__":
    # 一键运行，无命令行参数
    main()