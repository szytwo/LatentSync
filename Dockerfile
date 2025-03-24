# 使用 PyTorch 官方 CUDA 12.1 运行时镜像
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 安装编译依赖
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc g++ make \
    wget \
    autoconf \
    automake \
    cmake \
    git \
    libtool \
    pkg-config \
    yasm \
    nasm \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libfdk-aac-dev \
    libass-dev \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN gcc --version

# 从 Python 官网安装（推荐）
# 安装 Python 3.10.16 到自定义路径
RUN wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz && \
    tar -xzf Python-3.10.16.tgz && \
    cd Python-3.10.16 && \
    ./configure --prefix=/usr/local/python3.10.16 --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    rm -rf Python-3.10.16 Python-3.10.16.tgz

# 使用 update-alternatives 设置 Python 3.10.16 为默认 Python 版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/python3.10.16/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/python3.10.16/bin/pip3.10 1

# 验证 Python 和 pip 版本
RUN python --version && pip --version

# 下载并编译 FFmpeg
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && \
    ./configure \
    --prefix=/usr/local \
    --enable-gpl \
    --enable-nonfree \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libfdk-aac \
    --enable-libass \
    --enable-openssl \
    --enable-shared \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvenc \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ffmpeg

RUN ffmpeg --version

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
EXPOSE 22
EXPOSE 80
EXPOSE 7810

# 容器启动时执行 api.py
# CMD ["python", "api.py"]
