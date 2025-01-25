#!/bin/sh
set -x

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 데이터를 저장할 디렉토리
DATA_DIR="data/tiny_shakespeare"

# 디렉토리 생성
mkdir -p "$DATA_DIR"

# 파일 다운로드
echo "Downloading Tiny Shakespeare dataset..."
curl -o "$DATA_DIR/input.txt" "$URL"

# 다운로드 완료 메시지
if [ $? -eq 0 ]; then
    echo "Download complete. Data saved in '$DATA_DIR/input.txt'"
else
    echo "Download failed. Please check your internet connection or the URL."
fi
