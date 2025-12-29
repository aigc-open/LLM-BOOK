#!/bin/bash

# 视频生成脚本
# 注意：需要先将 drawio 文件导出为 PNG 图片

if [ -z "$OPENAI_API_KEY" ]; then
  echo "未设置 OPENAI_API_KEY，已退出。"
  exit 1
fi


cd "$(dirname "$0")"

# 目录配置
STATIC_DIR="videos/performance-tuning"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
CHAPTERS=(
  "01-封面.png:01-封面_script.txt:01-封面.mp4:封面"
  "02-概述.png:02-概述_script.txt:02-概述.mp4:概述"
  "03-ByTarget介绍.png:03-ByTarget介绍_script.txt:03-ByTarget介绍.mp4:ByTarget介绍"
  "04-ByTarget用法.png:04-ByTarget用法_script.txt:04-ByTarget用法.mp4:ByTarget用法"
  "05-架构标识.png:05-架构标识_script.txt:05-架构标识.mp4:架构标识"
  "06-内核参数配置.png:06-内核参数配置_script.txt:06-内核参数配置.mp4:内核参数配置"
  "07-矩阵乘法实战.png:07-矩阵乘法实战_script.txt:07-矩阵乘法实战.mp4:矩阵乘法实战"
  "08-latency参数.png:08-latency参数_script.txt:08-latency参数.mp4:latency参数"
  "09-allow_tma参数.png:09-allow_tma参数_script.txt:09-allow_tma参数.mp4:allow_tma参数"
  "10-性能提示策略.png:10-性能提示策略_script.txt:10-性能提示策略.mp4:性能提示策略"
  "11-调优策略矩阵.png:11-调优策略矩阵_script.txt:11-调优策略矩阵.mp4:调优策略矩阵"
  "12-内存密集型模式.png:12-内存密集型模式_script.txt:12-内存密集型模式.mp4:内存密集型模式"
  "13-计算密集型模式.png:13-计算密集型模式_script.txt:13-计算密集型模式.mp4:计算密集型模式"
  "14-向量加法实战.png:14-向量加法实战_script.txt:14-向量加法实战.mp4:向量加法实战"
  "15-最佳实践.png:15-最佳实践_script.txt:15-最佳实践.mp4:最佳实践"
  "16-总结.png:16-总结_script.txt:16-总结.mp4:总结"
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
  
  # 检查视频是否已存在
  if [ -f "$VIDEO_DIR/${video_file}" ]; then
    echo "⊙ ${title} 已存在，跳过生成"
    VIDEO_FILES+=("$VIDEO_DIR/${video_file}")
  else
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

