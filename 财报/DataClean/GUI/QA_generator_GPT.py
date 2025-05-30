import base64
import os
import time
from openai import OpenAI
import re
# ========= 配置 =========
OPENAI_API_KEY = "" #API key
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= 工具函数 =========

def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_qa_with_vision(base64_image, user_prompt=None):

    if user_prompt is None:
        user_prompt = (
            "You are an expert financial analyst specializing in annual report reading.\n\n"
            "Please carefully read and analyze the following **financial report page image**, and generate **10 high-quality Question-Answer (Q&A) pairs** based on the page content.\n\n"
            "The Q&A pairs must strictly follow these requirements:\n"
            "- **Content Scope**:\n"
            "  - Focus only on *factual financial information*, *management commentary*, *key business metrics*, *strategic trends*, and *compliance/legal information* found on the page.\n"
            "  - Avoid speculative questions or non-financial topics.\n"
            "- **Question Style**:\n"
            "  - Each question must be specific, clear, and directly related to a fact, metric, or strategy discussed in the page.\n"
            "  - Use formal business English without slang or vague expressions.\n"
            "- **Answer Style**:\n"
            "  - Answers must be concise, factual, and based only on visible information from the image.\n"
            "- **Formatting Requirements**:\n"
            "  - Use Markdown format.\n"
            "  - Each Q&A must be separated by \"---\".\n"
            "  - Each question should start with \"**Q:**\" and each answer with \"**A:**\".\n"
            "  - Example:\n\n"
            "    **Q:** What was Atlassian's total revenue for fiscal year 2024?  \n"
            "    **A:** Atlassian generated $4.3 billion in revenue for fiscal year 2024.\n\n"
            "- **Tone**:\n"
            "  - Formal, professional, and informative.\n\n"
            "Generate exactly 10 Q&A pairs.\n"
        )

    response = client.responses.create(
        model="gpt-4.1",
        temperature= 0 ,
        # temperature=0.3,
        # temperature=0.7,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
    )

    output_text = ""
    if response.output:
        for message in response.output:
            if message.role == "assistant":
                for content in message.content:
                    if content.type == "output_text":
                        output_text += content.text + "\n"
    return output_text.strip()

def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.png', filename)
    return int(match.group(1)) if match else float('inf')

def batch_process_images(folder_path, output_file="qa_results.txt", sleep_time=2):
    """批量处理一个文件夹下的所有图片，生成问答对并保存"""
    results = []

    # 判断传入的是单张图片还是文件夹
    if os.path.isdir(folder_path):
        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=extract_page_number
        )
    else:
        image_files = [os.path.basename(folder_path)]
        folder_path = os.path.dirname(folder_path)

    if not image_files:
        raise ValueError("No images found to process!")

    print(f" Found {len(image_files)} images, starting to generate Q&A...\n")

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(folder_path, img_name)
        print(f" [{idx}/{len(image_files)}] Processing image: {img_name}")

        try:
            image_base64 = encode_image(img_path)
            qa_text = generate_qa_with_vision(image_base64)

            results.append(f"Image: {img_name}\n{qa_text}\n{'='*100}\n")
            print(f" Completed: {img_name}\n")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}\n")

        time.sleep(sleep_time)  # 每张图片之间延迟防止API速率限制

    # 保存全部问答对到文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(item)

    print(f"\n All Q&A saved to {output_file}")

# ========= 主入口 =========

if __name__ == "__main__":
    screenshots_folder = "./screenshots/WoolworthsGroupAnnualReport2024/"  # 图片文件夹路径
    output_file = "screenshots/WoolworthsGroupAnnualReport2024/QA_Temp0/qa_generated_results.txt"  # 输出结果文件名
    batch_process_images(screenshots_folder, output_file)
