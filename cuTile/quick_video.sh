#!/bin/bash

# 视频生成脚本
# 注意：需要先将 drawio 文件导出为 PNG 图片

cd "$(dirname "$0")"

# 目录配置
STATIC_DIR="videos/data-model"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
CHAPTERS=(
  "01-封面.png:01-封面_script.txt:01-封面.mp4:封面"
  "02-概述.png:02-概述_script.txt:02-概述.mp4:概述"
  "03-全局数组.png:03-全局数组_script.txt:03-全局数组.mp4:全局数组"
  "04-Tile数组.png:04-Tile数组_script.txt:04-Tile数组.mp4:Tile数组"
  "05-空间概念.png:05-空间概念_script.txt:05-空间概念.mp4:空间概念"
  "06-形状广播.png:06-形状广播_script.txt:06-形状广播.mp4:形状广播"
  "07-数据类型.png:07-数据类型_script.txt:07-数据类型.mp4:数据类型"
  "08-类型转换.png:08-类型转换_script.txt:08-类型转换.mp4:类型转换"
  "09-标量与元组.png:09-标量与元组_script.txt:09-标量与元组.mp4:标量与元组"
  "10-舍入与填充.png:10-舍入与填充_script.txt:10-舍入与填充.mp4:舍入与填充"
  "11-实战示例.png:11-实战示例_script.txt:11-实战示例.mp4:实战示例"
  "12-总结.png:12-总结_script.txt:12-总结.mp4:总结"
)

# 语音配置
VOICE="nova"
SPEED="1.2"

# 创建视频输出目录
mkdir -p "$VIDEO_DIR"

echo "开始生成各章节视频..."
echo "================================"

# 用于存储所有视频路径（用于合并）
VIDEO_FILES=()

# 循环生成每个章节的视频
for chapter in "${CHAPTERS[@]}"; do
  # 分割章节信息
  IFS=':' read -r image_file script_file video_file title <<< "$chapter"
  
  echo "生成章节：${title}..."
  
  # 处理多个图片的情况（逗号分隔）
  # 将相对路径转换为绝对路径
  IFS=',' read -ra IMAGE_ARRAY <<< "$image_file"
  IMAGE_PATHS=()
  for img in "${IMAGE_ARRAY[@]}"; do
    IMAGE_PATHS+=("$STATIC_DIR/${img}")
  done
  # 用逗号连接所有图片路径
  IMAGE_INPUT=$(IFS=,; echo "${IMAGE_PATHS[*]}")
  
  python -m txt_images_to_ai_video generate \
    --input_txt="$STATIC_DIR/${script_file}" \
    --input_image="$IMAGE_INPUT" \
    --output_video="$VIDEO_DIR/${video_file}" \
    --voice="$VOICE" \
    --speed="$SPEED"
  
  # 检查视频是否生成成功
  if [ $? -eq 0 ]; then
    echo "✓ ${title} 生成成功"
    VIDEO_FILES+=("$VIDEO_DIR/${video_file}")
  else
    echo "✗ ${title} 生成失败"
    exit 1
  fi
  
  echo "--------------------------------"
done

# 合并所有视频
echo "合并所有视频..."
# 将数组转换为逗号分隔的字符串
VIDEO_INPUT=$(IFS=,; echo "${VIDEO_FILES[*]}")

python -m txt_images_to_ai_video merge_video \
  --input="$VIDEO_INPUT" \
  --output_video="$OUTPUT_DIR/final_video.mp4"

if [ $? -eq 0 ]; then
  echo "================================"
  echo "✓ 完成！最终视频已生成：$OUTPUT_DIR/final_video.mp4"
else
  echo "✗ 视频合并失败"
  exit 1
fi

