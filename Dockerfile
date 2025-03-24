# 使用 PyTorch 官方 CUDA 12.1 运行时镜像
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# 替换 Debian 软件源为清华镜像
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list

# 更新 APT 索引并安装必要的系统依赖：
# 直接从 Python 官网安装（推荐）
RUN apt-get update && apt-get install -y wget && \
    wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz && \
    tar -xzf Python-3.10.16.tgz && cd Python-3.10.16 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    rm -rf /var/lib/apt/lists/* Python-3.10.16 Python-3.10.16.tgz

# - build-essential：构建工具
# - ffmpeg：音视频处理工具
# - libgl1-mesa-glx & libglib2.0-0：部分图形和 OpenCV 相关库
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
# 环境变量
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.1 brand=nvidia,driver>=525,driver<600
ENV NV_CUDA_CUDART_VERSION=12.1.105-1
ENV CUDA_VERSION=12.1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-12-1

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 设置容器内工作目录为 /code
WORKDIR /code

# 将项目源代码复制到容器中
COPY . /code

# 升级 pip 并安装 Python 依赖：
# 1. 从本地 wheels 安装指定版本的 torch（避免网络下载大文件）
# 2. 根据 api_requirements.txt 安装其它依赖，使用阿里云镜像加速
# 3. 安装完成后删除 wheels 目录以减小镜像体积
RUN pip install --upgrade pip && \
    pip install -r api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露容器端口（这里设置为 7810，根据实际需求调整）
EXPOSE 7810

# 容器启动时执行 api.py
# CMD ["python", "api.py"]
