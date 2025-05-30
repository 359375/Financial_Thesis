import os
import time
import re
from PIL import Image
import google.generativeai as genai

# ========= 配置 =========
genai.configure(api_key="") #API KEY

# ========= 工具函数 =========

def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.png', filename)
    return int(match.group(1)) if match else float('inf')

def generate_qa_with_gemini(image_path, prompt_template=None):

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    img = Image.open(image_path)

    if prompt_template is None:
        prompt_template = (
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
    try:
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [prompt_template, img]}
            ],
            generation_config={
                "temperature": 0.7
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def batch_process_images_with_gemini(folder_path, output_file="qa_results_gemini.txt", sleep_time=2):
    results = []

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

    print(f"Found {len(image_files)} images, starting Gemini Q&A generation...\n")

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(folder_path, img_name)
        print(f"[{idx}/{len(image_files)}] Processing image: {img_name}")

        qa_text = generate_qa_with_gemini(img_path)
        results.append(f"Image: {img_name}\n{qa_text}\n{'='*100}\n")

        time.sleep(sleep_time)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(item)

    print(f"\nAll Gemini Q&A saved to {output_file}")

# ========= 主入口示例 =========
if __name__ == "__main__":
    screenshots_folder = "./screenshots/WoolworthsGroupAnnualReport2024/"
    output_file = "screenshots/WoolworthsGroupAnnualReport2024/QA_Temp07_gemini/qa_gemini_temp07.txt"
    batch_process_images_with_gemini(screenshots_folder, output_file)
