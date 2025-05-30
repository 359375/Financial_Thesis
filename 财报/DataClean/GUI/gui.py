import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import shutil
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import re
from openai import OpenAI

UPLOAD_DIR = "./uploaded_reports"
SCREENSHOT_DIR = "./screenshots"
DEFAULT_PAGE_OFFSET = 0
client = OpenAI(api_key="API KEY")  # API KEY


# ========= 工具函数 =========
def extract_first_pages(pdf_path, num_pages=10):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(num_pages, len(doc))):
        text += doc[i].get_text()
    return text

def gpt_check_if_financial_report(text):
    prompt = f"""
You are a professional analyst of corporate financial reports. You are responsible for determining whether a PDF document is a complete and usable “company annual report”.  You are also responsible for determining whether the report contains a clear, structured chapter table of contents. The document is in English and consists of subject matter. At the time of the determination, you must strictly adhere to the following criteria, not to imagine, infer or make up the information on their own.

Task 1: Is it a “company's annual financial report”?
Please determine whether at least two of the following conditions are met:
1. the title or body of the document contains the term “Annual Report” or its equivalent.  
2. Clearly refer to summary information for the “entire fiscal year” (e.g., “2023 fiscal year summary”).  
3. The text contains multiple sections such as business units, financial indicators, management speeches, etc. (complete and clearly structured).
4. The document contains at least two representative chapter titles, e.g., “Financial Statements”, “Corporate Governance”, “Business Analysis”, “Risk Management”. Risk Management”.  

Quarterly newsletters, interim announcements, shareholder notices, press releases, and summary pages are not part of the annual financial report and should be labeled as “non-financial”.

Task 2: Does it contain a “structured table of contents page”?
Determine if there is a “structured section table of contents page” (the table of contents page must be used for quick navigation).

 At least three lines with the following structure:
  - Clear section title
  - Clear page numbers (numeric)
  - Clear section headings Clear page numbers (numeric) Visual alignment structure between headings and page numbers (e.g., dotted lines, spaces, or indentation)
 Consistent structure for each line, e.g.:
  - “Chapter 1 Introduction to the company .................... 1”
  - “III Financial Statements ........................ 25”
  - “5. Risk Factors ........................ 30”

 All table of contents entries should appear clearly at the beginning of the document (within the first 10 pages).
 If there are no page numbers, or if the structure is not uniform, or if there are simply paragraph headings, please consider “No Table of Contents”.
 If the GPT is unable to verify that the table of contents is structured (e.g., formatting is ambiguous or content is missing), prefer “No Table of Contents”.

Output Requirements
Please return strictly only one of the following three categories, without explanation:

- Continuing analysis
- Non-Annual Financial Reports
- No Table of Contents
Below are the first few pages of the document uploaded by the user:
{text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
def gpt_analyze_page_ranges(text):
    prompt = f"""
    You are a senior financial analyst specializing in company annual financial reports. Below is the Table of Contents (TOC) and Introduction sections of a company's annual report.
    Your task is to help identify a range of no more than 4 non-overlapping screenshots based solely on the section headings and page numbers listed in the Table of Contents (TOC). 
    These screenshots will be used for training on the Financial Documentation Q&A model.
    Please strictly adhere to the following rules:
    1. Use only items from the Table of Contents. Ignore all other descriptive content or text.
    2. Scope the section:
       Let `start_page` = the page number of the section listed
       Let `end_page` = (page number of next section) - 1
       If no section follows, assume that the section ends on page 250.
    3. Do not try to guess the section length. Do not merge sections across boundaries unless they are listed consecutively in the TOC.
    4. Prioritize chapters that fall into the following categories:
       A. Executive commentary (e.g., CEO commentary, Chairman's report, CFO commentary)
       B. Financial highlights, key performance indicators or summary tables
       C. Core financial statements and notes (e.g., income statement, balance sheet, cash flow statement)
       D. business segment or product line performance (e.g., iron ore, retail)
       E. charts or tables reflecting year-over-year trends or comparisons
       (Optional: add 1 ESG, risk or governance scope if relevant)
    Use this format to return your output:
    Suggested Screenshot Scope:
    6-8, 16-20, 83-85, 132-140
    Below is the table of contents and introduction to the document:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def parse_page_ranges(gpt_text):
    ranges = []


    matches = re.findall(r"(?<!\d)(\d{1,3})\s*[-–]\s*(\d{1,3})(?!\d)", gpt_text)
    for start, end in matches:
        ranges.append((int(start), int(end)))


    lines = gpt_text.splitlines()
    for line in lines:
        if re.match(r"^\d+\.", line.strip()):
            continue
        singles = re.findall(r"\b\d{1,3}\b", line)
        for s in singles:
            page = int(s)
            if not any(start <= page <= end for start, end in ranges):
                ranges.append((page, page))
    ranges = sorted(list(set(ranges)))
    return ranges



def adjust_page_ranges(page_ranges, offset=0):
    return [(start + offset, end + offset) for start, end in page_ranges]

def screenshot_pages(pdf_path, page_ranges, output_folder, offset):
    os.makedirs(output_folder, exist_ok=True)
    for start, end in page_ranges:
        images = convert_from_path(pdf_path, dpi=200, first_page=start, last_page=end)
        for i, img in enumerate(images, start=start):
            img.save(os.path.join(output_folder, f"page_{i-offset}.png"))


class ReportUploaderApp:
    def __init__(self, master):
        self.master = master
        master.title("Intelligent Financial Report Q&A Generator")
        master.geometry("700x600")
        master.configure(bg="#ffffff")

        self.page_offset = tk.IntVar(value=DEFAULT_PAGE_OFFSET)
        self.double_page_mode = tk.BooleanVar(value=False)

        self.build_header(master)
        self.build_upload_section(master)
        self.build_file_list_section(master)
        self.build_offset_input(master)
        self.build_process_button(master)

        self.refresh_file_list()

    def build_header(self, master):
        header = tk.Frame(master, bg="#004c99")
        header.pack(fill=tk.X)
        title = tk.Label(header, text="Financial report upload and management tools", font=("Helvetica", 20, "bold"), fg="white", bg="#004c99", pady=20)
        title.pack()

    def build_upload_section(self, master):
        section = tk.Frame(master, bg="#ffffff")
        section.pack(pady=15)
        upload_btn = ttk.Button(section, text="Upload PDF Financial Report", command=self.upload_pdf)
        upload_btn.grid(row=0, column=0, padx=10)
        self.status_label = tk.Label(section, text="", bg="#ffffff", fg="green", font=("Helvetica", 10))
        self.status_label.grid(row=0, column=1, sticky="w")

    def build_file_list_section(self, master):
        frame = tk.Frame(master, bg="#ffffff")
        frame.pack(pady=10, fill=tk.BOTH, expand=True)
        label = tk.Label(frame, text="Currently uploaded documents：", font=("Helvetica", 12, "bold"), bg="#ffffff")
        label.pack(anchor="w", padx=15)
        self.file_listbox = tk.Listbox(frame, font=("Courier", 11), height=8, bg="#f2f2f2", selectbackground="#cce5ff")
        self.file_listbox.pack(fill=tk.BOTH, padx=15, pady=5, expand=True)

    def build_offset_input(self, master):
        frame = tk.Frame(master, bg="#ffffff")
        frame.pack(pady=5)
        label = tk.Label(frame, text="Page offset (e.g. +2):", font=("Helvetica", 10), bg="#ffffff")
        label.pack(side=tk.LEFT)
        self.offset_entry = ttk.Entry(frame, textvariable=self.page_offset, width=5)
        self.offset_entry.pack(side=tk.LEFT, padx=5)

        self.double_page_checkbox = ttk.Checkbutton(
            frame, text="double-page mode", variable=self.double_page_mode
        )
        self.double_page_checkbox.pack(side=tk.LEFT, padx=20)

    def build_process_button(self, master):
        btn_frame = tk.Frame(master, bg="#ffffff")
        btn_frame.pack(pady=10)
        select_btn = ttk.Button(btn_frame, text="Analyze and take screenshots of selected financial reports", command=self.select_report)
        select_btn.pack()

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF 文件", "*.pdf")])
        if not file_path:
            return
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filename = os.path.basename(file_path)
        target_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(target_path):
            messagebox.showwarning("Duplicate file", f"{filename} 已存在。")
            self.status_label.config(text="file Existing", fg="orange")
        else:
            shutil.copy(file_path, target_path)
            self.status_label.config(text="Upload success", fg="green")
        self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
        for f in pdf_files:
            self.file_listbox.insert(tk.END, f)

    def select_report(self):
        try:
            index = self.file_listbox.curselection()[0]
            filename = self.file_listbox.get(index)
            full_path = os.path.join(UPLOAD_DIR, filename)

            # Step 1
            text = extract_first_pages(full_path, num_pages=5)

            # Step 2
            status = gpt_check_if_financial_report(text)
            print(status)
            if "Continuing analysis" not in status:
                messagebox.showwarning("X Non-financial reports", f"GPT Audit Results：\n{status}")
                return

            # Step 3
            gpt_reply = gpt_analyze_page_ranges(text)
            print(gpt_reply)
            raw_ranges = parse_page_ranges(gpt_reply)
            print(raw_ranges)


            if self.double_page_mode.get():
                offset = int(self.offset_entry.get())


                double_mapped_ranges = []
                for start, end in raw_ranges:


                    real_start = (start + offset ) // 2
                    real_end = (end + offset ) // 2

                    double_mapped_ranges.append((real_start, real_end))

                adjusted_ranges = double_mapped_ranges
                print(f"adjusted_ranges {adjusted_ranges}")
            else:

                offset = int(self.offset_entry.get())
                adjusted_ranges = adjust_page_ranges(raw_ranges, offset)

            output_folder = os.path.join(SCREENSHOT_DIR, filename.replace(".pdf", ""))

            screenshot_pages(full_path, adjusted_ranges, output_folder, offset)

            messagebox.showinfo(" Completed", f"Recommended：\n{gpt_reply}\n\n Pics to：\n{output_folder}")

        except IndexError:
            messagebox.showwarning("No selection", "Please select one file.")
        except Exception as e:
            messagebox.showerror("Wrong", f"Mistake：\n{str(e)}")

# ========= 启动 GUI =========
if __name__ == "__main__":
    root = tk.Tk()
    app = ReportUploaderApp(root)
    root.mainloop()