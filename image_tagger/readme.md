# readme

- 项目参考: "https://github.com/neggles/wdv3-timm"
- 模型: "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3"

## 作用

给图片打标的。直接 `python tagger.py` 即可。

```python
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

```

```
└── models--SmilingWolf--wd-eva02-large-tagger-v3
    ├── blobs
    ├── refs
    └── snapshots
        └── b25b82a03f7282e41aa2f257a52c7583b710bd1c
            ├── .gitattributes
            ├── config.json
            ├── model.msgpack
            ├── model.onnx
            ├── model.safetensors
            ├── README.md
            ├── selected_tags.csv
            └── sw_jax_cv_config.json
```