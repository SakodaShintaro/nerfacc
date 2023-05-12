FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get install -y git

# WORKDIRを設定
WORKDIR /root

# dotfileの導入(主にvimのため)
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh

