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
STATIC_DIR="vllm/simple"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
# 注意同步本地静态资源目录中的文件名和格式
CHAPTERS=(
  "01_cover.png:01_cover_script.txt:01_cover.mp4:封面"
  "02_intro.png:02_intro_script.txt:02_intro.mp4:介绍"
  "03_paged_attention.png:03_paged_attention_script.txt:03_paged_attention.mp4:分页注意力"
  "04_advantages.png:04_advantages_script.txt:04_advantages.mp4:优势"
  "05_installation.png:05_installation_script.txt:05_installation.mp4:安装"
  "06_model_support.png:06_model_support_script.txt:06_model_support.mp4:模型支持"
  "07_usage_openai_api.png:07_usage_openai_api_script.txt:07_usage_openai_api.mp4:OpenAI API使用"
  "08_usage_native_api.png:08_usage_native_api_script.txt:08_usage_native_api.mp4:原生API使用"
  "09_usage_langchain.png:09_usage_langchain_script.txt:09_usage_langchain.mp4:LangChain使用"
  "10_parameters.png:10_parameters_script.txt:10_parameters.mp4:参数配置"
  "11_examples.png:11_examples_script.txt:11_examples.mp4:示例"
  "12_best_practices.png:12_best_practices_script.txt:12_best_practices.mp4:最佳实践"
  "13_summary.png:13_summary_script.txt:13_summary.mp4:总结"
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

