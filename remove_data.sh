#!/bin/bash

# result 디렉토리 경로
RESULT_DIR="result"

# 대상 폴더 리스트
TARGET_FOLDERS=("dist" "energy" "sample")

# 삭제 제외 패턴
EXCLUDE_PATTERNS=("GT_0" "Teacher_0" "Student")

# 각 대상 폴더 순회
for dir in "$RESULT_DIR"/*; do
    for folder in "${TARGET_FOLDERS[@]}"; do
        TARGET_DIR="$dir/$folder"
        if [ -d "$TARGET_DIR" ]; then
            echo "Processing directory: $TARGET_DIR"
            for file in "$TARGET_DIR"/*; do
                # 파일 이름에 EXCLUDE_PATTERNS 중 하나라도 포함되어 있는지 확인
                if [[ -f "$file" && ! "$(basename "$file")" =~ ${EXCLUDE_PATTERNS[0]}|${EXCLUDE_PATTERNS[1]}|${EXCLUDE_PATTERNS[2]} ]]; then
                    echo "Deleting $file"
                    rm "$file"
                fi
            done
        fi
    done
done