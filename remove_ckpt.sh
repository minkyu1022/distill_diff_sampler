#!/bin/bash

# result 디렉토리 경로
RESULT_DIR="result"

# result 디렉토리 내 모든 하위 디렉토리 순회
for dir in "$RESULT_DIR"/*/; do
    # ckpt 디렉토리 확인
    CKPT_DIR="${dir}ckpt"
    if [ -d "$CKPT_DIR" ]; then
        # ckpt 디렉토리 내 파일 목록 가져오기 (정렬)
        files=($(ls -v "$CKPT_DIR"/*.pth 2>/dev/null))
        
        # 파일이 2개 이상인 경우 마지막 파일 제외하고 삭제
        if [ ${#files[@]} -gt 1 ]; then
            unset files[-1] # 마지막 파일 제외
            for file in "${files[@]}"; do
                echo "Deleting $file"
                rm "$file"
            done
        fi
    fi
done