import re
import pandas as pd
import re
import csv
# 文件路径
input_file ="./screenshots/WoolworthsGroupAnnualReport2024/QA_Temp0/qa_generated_results.txt"
output_file= "screenshots/WoolworthsGroupAnnualReport2024/QA_Temp0/wws_qa_output.csv"
rows = []
current_image = "unknown"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    # 检测新页面
    if line.startswith("Image:"):
        current_image = line.replace("Image:", "").strip()

    # 匹配问答对
    if line.startswith("**Q:**"):
        question = line.replace("**Q:**", "").strip()
        i += 1
        # 往下找 answer
        while i < len(lines) and not lines[i].strip().startswith("**A:**"):
            question += " " + lines[i].strip()
            i += 1
        if i < len(lines):
            answer_line = lines[i].strip()
            answer = answer_line.replace("**A:**", "").strip()
            i += 1
            # 多行 answer 拼接
            while i < len(lines) and not lines[i].strip().startswith("**Q:**") and not lines[i].strip().startswith("Image:") and not lines[i].strip().startswith("==="):
                if lines[i].strip().startswith("---"):
                    i += 1
                    continue
                answer += " " + lines[i].strip()
                i += 1
            rows.append([current_image, question, answer])
        else:
            i += 1
    else:
        i += 1

# 写入 CSV
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Page', 'Question', 'Answer'])
    writer.writerows(rows)

print(f" Succsee：Total {len(rows)} QAs，Save to {output_file}")