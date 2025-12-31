#!/bin/bash

# 视频生成脚本
# 注意：需要先将 drawio 文件导出为 PNG 图片

if [ -z "$OPENAI_API_KEY" ]; then
  echo "未设置 OPENAI_API_KEY，已退出。"
  exit 1
fi


cd "$(dirname "$0")"

# 目录配置
STATIC_DIR="agents/ph8/videos"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
# 注意同步本地静态资源目录中的文件名和格式
CHAPTERS=(
  "01-cover.png:01-cover_script.txt:01-cover.mp4:封面"
  "02-features.png:02-features_script.txt:02-features.mp4:为什么选择CodeX"
  "03-homepage.png:03-homepage_script.txt:03-homepage.mp4:ph8大模型密钥获取"
  "04-model-square.png:04-model-square_script.txt:04-model-square.mp4:ph8模型选择"
  "05-vendors.png:05-vendors_script.txt:05-vendors.mp4:快速配置"
  "06-models.png:06-models_script.txt:06-models.mp4:主界面介绍"
  "07-api.png:07-api_script.txt:07-api.mp4:两大工作模式"
  "08-cases.png:08-cases_script.txt:08-cases.mp4:五大实战场景"
  "09-advantages.png:09-advantages_script.txt:09-advantages.mp4:提示词工程"
  "10-start.png:10-start_script.txt:10-start.mp4:成功案例"
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

