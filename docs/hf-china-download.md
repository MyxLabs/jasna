# 中国地区 Hugging Face 模型加速下载

脚本：`scripts/download_hf_cn.py`

默认从 **https://hf-mirror.com** 下载 Jasna BasicVSR++ 免费模型至 `model_weights/`。
固定模型修订版本并校验 SHA-256，重复运行时检查已有文件；底层
`huggingface_hub` 使用缓存和断点续传机制。镜像属于第三方服务，
不是 Hugging Face 官方网站，可靠性、可用性和安全性不能保证。
请勿向不可信的镜像发送访问令牌；脚本仅允许使用 `--official --use-token`。

## 安装依赖（AutoDL Python 3.10）

```bash
/usr/bin/python3 -m pip install "huggingface_hub>=0.25,<1"
```

不需要升级 torch / torchvision / torch_musa。

## 一键下载 Jasna BasicVSR++ 权重

```bash
cd ~/jasna && /usr/bin/python3 scripts/download_hf_cn.py
```

预期目标文件：`model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`。

使用 HF 官方源重试：

```bash
cd ~/jasna && /usr/bin/python3 scripts/download_hf_cn.py --official
```

如果镜像的 Xet token 服务不可用：

```bash
cd ~/jasna && /usr/bin/python3 scripts/download_hf_cn.py --disable-xet
```

或指定兼容的 HTTPS 镜像：`--endpoint https://your-hf-mirror.example`。

## 下载任意 Hugging Face 模型

下载单文件：

```bash
/usr/bin/python3 scripts/download_hf_cn.py --repo org/model --file model.safetensors --revision main
```

整仓库下载：

```bash
/usr/bin/python3 scripts/download_hf_cn.py --repo org/model --snapshot --workers 4
```

注意：**任意仓库或非默认修订版不会自动获得 SHA-256 校验**。
如掌握可信的文件摘要，可添加 `--sha256 64位十六进制摘要`。

私有仓库访问需要选择官方源，并在安全的环境变量 `HF_TOKEN` 中配置
自己的访问令牌，再传入 `--official --use-token`。不要将访问令牌写到
命令行、提交日志或 GitHub 仓库。

此脚本负责模型下载，并不解决 MTT S4000 的 GPU 算子兼容性；
BasicVSR++ 推理测试参见 `scripts/check_musa_basicvsrpp.py`。
