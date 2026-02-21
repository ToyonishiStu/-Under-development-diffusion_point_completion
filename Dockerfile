# ベースイメージ（CUDA 12.8 + Ubuntu 22.04）
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 非対話モード（インストール時の質問をスキップ）
ENV DEBIAN_FRONTEND=noninteractive

# 基本ツールのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python のパス統一
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# 依存パッケージのインストール（必要なら requirements.txt を追加）
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ワークディレクトリ
WORKDIR /workspaces

# デフォルトコマンド
CMD ["/bin/bash"]
