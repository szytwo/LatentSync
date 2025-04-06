## 安装

```
conda create --prefix ./venv python==3.10.13

conda activate ./venv

pip install -r ./api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

docker build -t latentsync1.5:latest .  # 构建镜像
docker load -i latentsync1.5.tar # 导入镜像
docker save -o latentsync1.5.tar latentsync1.5:latest # 导出镜像
docker-compose up -d # 后台运行容器

```

## 下载模型

```

https://huggingface.co/ByteDance/LatentSync
https://huggingface.co/ByteDance/LatentSync-1.5
https://huggingface.co/stabilityai/sd-vae-ft-mse

```

Triton2.0-Windows链接 ：https://pan.quark.cn/s/f98cac8c8375
提取码：HLW5

官网 https://github.com/bytedance/LatentSync