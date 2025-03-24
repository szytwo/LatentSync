# 基于 Python 3.10.16-slim 镜像（Debian Bullseye 精简版）
FROM python:3.10.16-slim

# 更新 APT 索引并安装必要的系统依赖：
# - build-essential：构建工具
# - ffmpeg：音视频处理工具
# - libgl1-mesa-glx & libglib2.0-0：部分图形和 OpenCV 相关库
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0
RUN rm -rf /var/lib/apt/lists/*

# 设置容器内工作目录为 /code
WORKDIR /code

# 复制依赖清单和本地 wheel 文件到容器中
COPY api_requirements.txt /code/api_requirements.txt
COPY wheels/torch-2.2.2+cu121-cp310-cp310-linux_x86_64.whl /wheels/torch-2.2.2+cu121-cp310-cp310-linux_x86_64.whl

# 升级 pip 并安装 Python 依赖：
# 1. 从本地 wheels 安装指定版本的 torch（避免网络下载大文件）
# 2. 根据 api_requirements.txt 安装其它依赖，使用阿里云镜像加速
# 3. 安装完成后删除 wheels 目录以减小镜像体积
RUN pip install --upgrade pip
RUN pip install --find-links=/wheels torch==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN rm -rf /wheels

# 将项目源代码复制到容器中
COPY . /code

# 暴露容器端口（这里设置为 7810，根据实际需求调整）
EXPOSE 7810

# 容器启动时执行 api.py
CMD ["python", "api.py"]
