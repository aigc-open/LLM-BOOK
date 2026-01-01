#!/bin/bash

source env.sh

# 视频生成脚本
# 注意：需要先将 drawio 文件导出为 PNG 图片

if [ -z "$OPENAI_API_KEY" ]; then
  echo "未设置 OPENAI_API_KEY，已退出。"
  exit 1
fi


cd "$(dirname "$0")"

# 目录配置
STATIC_DIR="agents/maxkb/videos"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
# 注意同步本地静态资源目录中的文件名和格式
CHAPTERS=(
  "page01-cover.drawio.png:page01-cover-script.txt:page01-cover.mp4:封面"
  "page02-problems.drawio.png:page02-problems-script.txt:page02-problems.mp4:问题分析"
  "page03-rag-intro.drawio.png:page03-rag-intro-script.txt:page03-rag-intro.mp4:RAG介绍"
  "page04-rag-workflow1.drawio.png:page04-rag-workflow1-script.txt:page04-rag-workflow1.mp4:RAG工作流程1"
  "page05-rag-workflow2.drawio.png:page05-rag-workflow2-script.txt:page05-rag-workflow2.mp4:RAG工作流程2"
  "page06-rag-comparison.drawio.png:page06-rag-comparison-script.txt:page06-rag-comparison.mp4:RAG对比"
  "page07-maxkb-intro.drawio.png:page07-maxkb-intro-script.txt:page07-maxkb-intro.mp4:MaxKB介绍"
  "page08-deployment.drawio.png:page08-deployment-script.txt:page08-deployment.mp4:部署方案"
  "page09-maxkb-01.png:page09-maxkb-01-script.txt:page09-maxkb-01.mp4:MaxKB演示1"
  "page09-maxkb-02.png:page09-maxkb-02-script.txt:page09-maxkb-02.mp4:MaxKB演示2"
  "page09-maxkb-03.png:page09-maxkb-03-script.txt:page09-maxkb-03.mp4:MaxKB演示3"
  "page09-maxkb-04.png:page09-maxkb-04-script.txt:page09-maxkb-04.mp4:MaxKB演示4"
  "page09-maxkb-05.png:page09-maxkb-05-script.txt:page09-maxkb-05.mp4:MaxKB演示5"
  "page09-maxkb-06.png:page09-maxkb-06-script.txt:page09-maxkb-06.mp4:MaxKB演示6"
  "page09-maxkb-07.png:page09-maxkb-07-script.txt:page09-maxkb-07.mp4:MaxKB演示7"
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

