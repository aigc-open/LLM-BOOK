#!/bin/bash

# 视频生成脚本
# 注意：需要先将 drawio 文件导出为 PNG 图片

if [ -z "$OPENAI_API_KEY" ]; then
  echo "未设置 OPENAI_API_KEY，已退出。"
  exit 1
fi


cd "$(dirname "$0")"

# 目录配置
STATIC_DIR="agents/kilo_code/videos"          # 静态资源目录（输入：图片、旁白文本）
OUTPUT_DIR="$STATIC_DIR/output"          # 输出目录（生成的视频）
VIDEO_DIR="$OUTPUT_DIR/videos"  # 章节视频目录

# 配置章节列表（格式：图片文件名:旁白文件名:视频文件名:章节标题）
# 如需调整章节和资源，请同步修改本地静态资源目录中的文件名
CHAPTERS=(
  "00_封面.png:00_封面_script.txt:00_封面.mp4:封面"
  "01-kilo安装.png:01-kilo安装_script.txt:01-kilo安装.mp4:kilo安装"
  "02-ph8大模型密钥获取.png:02-ph8大模型密钥获取_script.txt:02-ph8大模型密钥获取.mp4:ph8大模型密钥获取"
  "03-配置密钥供应商.png:03-配置密钥供应商_script.txt:03-配置密钥供应商.mp4:配置密钥供应商"
  "04-kilo配置系统读写mcp等权限.png:04-kilo配置系统读写mcp等权限_script.txt:04-kilo配置系统读写mcp等权限.mp4:kilo配置系统读写mcp等权限"
  "05-chat效果.png:05-chat效果_script.txt:05-chat效果.mp4:chat效果"
  "06_提示词编写原则.png:06_提示词编写原则_script.txt:06_提示词编写原则.mp4:提示词编写原则"
  "07_提示词实战模板.png:07_提示词实战模板_script.txt:07_提示词实战模板.mp4:提示词实战模板"
  "08_如何改代码.png:08_如何改代码_script.txt:08_如何改代码.mp4:如何改代码"
  "09_代码修改示例.png:09_代码修改示例_script.txt:09_代码修改示例.mp4:代码修改示例"
  "10_使用技巧总结.png:10_使用技巧总结_script.txt:10_使用技巧总结.mp4:使用技巧总结"
  "11_效率提升对比.png:11_效率提升对比_script.txt:11_效率提升对比.mp4:效率提升对比"
  "12_结尾.png:12_结尾_script.txt:12_结尾.mp4:结尾"
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

