import streamlit as st
import ollama
import fitz  # PyMuPDF
from opencc import OpenCC
import re
import os 

from dotenv import load_dotenv
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

print(LLM_MODEL)
def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
# --- 函數定義 ---
# 初始化轉換器（s2t 表示簡體轉繁體）
cc = OpenCC('s2t')

def enforce_traditional(text):
    return cc.convert(text)

# 讀取 PDF 檔案文字（不進行 OCR）
def get_pdf_text(pdf_files):
    """
    從多個 PDF 檔案中提取文字。
    """
    full_text = ""
    for pdf_file in pdf_files:
        st.write(f"正在處理檔案：{pdf_file.name}")
        try:
            # 使用 PyMuPDF 從 Streamlit 的上傳檔案中讀取
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # 提取頁面文字
                page_text = page.get_text()
                full_text += page_text + "\n"
                
        except Exception as e:
            st.error(f"處理檔案 {pdf_file.name} 時發生錯誤：{e}")
            
    return full_text

# 使用 Ollama 總結文字
def get_ollama_summary(text):
    """
    將提取的文字傳送給 Ollama 模型進行總結。
    
    此函式已優化提示詞結構，明確設定角色與摘要規則，
    以強制模型產出繁體中文且符合指定格式的系統操作流程摘要。
    
    Args:
        text (str): 待總結的文字內容。
        
    Returns:
        str: 模型的總結結果，若發生錯誤則返回錯誤訊息。
    """
    if not text or not text.strip():
        return "沒有可總結的文字。"

    # 設定模型的通用行為，強調角色和語言限制
    system_prompt = """
    你將擔任專業的文件摘要專家。
    請依據使用者提供的規則，將內容整理成摘要。
    你只能用繁體中文輸出。
    """

    # 包含所有具體摘要規則的提示詞
    user_prompt = f"""
    請閱讀我提供的PDF文件（公司ERP系統操作手冊），並依據以下所有規則，將內容整理成一份簡潔、有條理的系統操作流程摘要。

    ### 輸出規則 ###
    1. **核心目標：**
       - 僅專注於「系統操作流程」。

    2. **格式要求：**
       - 採用「條列式」呈現。
       - 每頁摘要 10 到 15 個關鍵重點。

    3. **內容要求：**
       - 每個重點字數不超過 500 字。
       - 保留關鍵專有名詞與重要數字。
       - 避免重複內容。
       - 忽略與操作流程無關的所有細節與背景資訊。

    ### 待處理文件內容 ###
    {text[:100000]}
    """

    try:
        response = ollama.chat(
            # model='qwen2:7b',  
            # model='qwen3:8b',
            model=LLM_MODEL,

            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"與 Ollama 溝通時發生錯誤：{e}"

# --- Streamlit 介面 ---
st.set_page_config(page_title="LLM PDF 總結器", layout="wide")
st.header("使用 LLM 進行 PDF 總結 (僅文字)")
st.info("此程式會提取 PDF 中的文字，然後交由 LLM 總結。")

# 上傳多個 PDF 檔案
pdf_files = st.file_uploader(
    "請上傳你的 PDF 檔案", type=["pdf"], accept_multiple_files=True
)

if pdf_files and st.button("開始總結", key="summarize_button"):
    with st.spinner("正在讀取 PDF 內容..."):
        full_document_text = get_pdf_text(pdf_files)
        
    # 顯示提取出的文字（方便除錯）
    with st.expander("點此查看所有提取出的文字內容"):
        st.text_area("所有文件文字內容", value=full_document_text, height=500)
        
    if full_document_text.strip():
        with st.spinner("正在使用 LLM 總結文件..."):
            summary = get_ollama_summary(full_document_text)
        st.success("總結完成！")
        st.subheader("文件總結")
        # 強制轉繁體中文
        summary = enforce_traditional(summary)
        summary = remove_think_tags(summary) 
        st.write(summary)
    else:
        st.warning("沒有可總結的文字，請確保 PDF 檔案內容可被提取。")
