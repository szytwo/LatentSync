# 使用 Python 3.10.13 基于 Debian Bullseye 的精简版镜像
FROM python:3.10.13-slim-bullseye

# 安装系统依赖：构建工具、ffmpeg（用于音视频处理）、以及 opencv 等可能需要的库
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /code

# 复制 requirements.txt 到容器中
COPY requirements.txt /code/requirements.txt

# 升级 pip 并安装 Python 依赖（其中包含额外的 PyTorch 索引）
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# 将整个项目代码复制到容器中
COPY . /code

# 如果项目有对外提供服务（例如 Gradio demo），可以暴露相应端口，这里以 7860 为例
EXPOSE 7810

# 设置默认启动命令（根据你的项目入口文件调整）
CMD ["python", "api.py"]
