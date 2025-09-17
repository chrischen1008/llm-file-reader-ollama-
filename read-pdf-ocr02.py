import streamlit as st
import ollama
import fitz # PyMuPDF
import io
import pytesseract
from PIL import Image
from opencc import OpenCC
# --- 函數定義 ---
# 設定 pytesseract 的安裝路徑
# 這是 Windows 使用者可能需要的步驟，若在其他作業系統上可省略
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    st.warning("無法自動設定 Tesseract-OCR 路徑，請確認其已正確安裝並在系統 PATH 中。")

# 初始化轉換器（s2t 表示簡體轉繁體）
cc = OpenCC('s2t')

def enforce_traditional(text):
    return cc.convert(text)
    
# 讀取 PDF 檔案，同時進行文字提取和圖片 OCR
def get_pdf_content_with_ocr(pdf_files):
    """
    從多個 PDF 檔案中提取文字和圖片內容。
    對於圖片，它會使用 Tesseract 進行 OCR 辨識，並將結果加入到文件中。
    """
    full_text = ""
    for pdf_file in pdf_files:
        st.write(f"正在處理檔案：{pdf_file.name}")
        try:
            # 使用 PyMuPDF 從 Streamlit 的上傳檔案中讀取
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # 1. 提取頁面上的文字
                page_text = page.get_text()
                full_text += page_text + "\n"
                
                # 2. 提取頁面上的圖片並進行 OCR
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 將圖片資料轉換為 PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # 進行 OCR，指定繁體中文和英文語言
                    ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra+eng')
                    if ocr_text.strip():
                        full_text += f"\n[圖片內容 OCR 辨識結果 (第 {page_num+1} 頁, 圖片 {img_index+1})]:\n{ocr_text}\n"
                        
        except Exception as e:
            st.error(f"處理檔案 {pdf_file.name} 時發生錯誤：{e}")
            
    return full_text

# 使用 Ollama 總結文字
def get_ollama_summary(text):
    """
    將提取的文字傳送給 Ollama 模型進行總結（強制繁體中文）。
    """
    if not text.strip():
        return "沒有可總結的文字。"

    system_prompt = """
                    ###你只能用繁體中文輸出###
                    根據使用者所提供的內文依以下規則完成整理重點：
                    - 將內容整理產出ERP系統的操作手冊
                    - 保留關鍵專有名詞與重要數字
                    - 全文使用繁體中文
                    """

    # 將文件內容包裝成強化 prompt
    prompt = (
        f"{system_prompt}\n\n"
        f"""請閱讀我上傳的PDF文件,文件是公司ERP系統的操作手冊，依以下規則整理重點：
            1. 摘要目標：
               - 產出ERP系統的操作手冊

            2. 摘要規則：
               - 每頁整理 3到5 個重點
               - 每個重點不超過 100 字
               - 保留關鍵專有名詞與重要數字

            3. 輸出格式：
               - 使用 Markdown 表格
               - 欄位：頁碼｜重點摘要
               - 全文使用繁體中文

            4. 其他：
               - 避免重複內容
               - 省略與主題無關的細節\n\n"""
        f"{text[:30000]}"
    )

    try:
        response = ollama.chat(
            #model='gemma3:12b',
            model='qwen2:7b',#偶爾會出現簡體中文
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"與 Ollama 溝通時發生錯誤。請確認 Ollama 服務已啟動且模型 'gemma3:latest' 已下載。錯誤訊息：{e}"


# --- Streamlit 介面 ---
st.set_page_config(page_title="LLM PDF 總結器 (支援圖片)", layout="wide")
st.header("使用 LLM 進行 PDF 總結 (包含圖片 OCR)")
st.info("此程式會提取 PDF 中的文字，並對圖片進行 OCR 辨識，然後一併交由 LLM 總結。")

# 上傳多個 PDF 檔案
pdf_files = st.file_uploader(
    "請上傳你的 PDF 檔案", type=["pdf"], accept_multiple_files=True
)

if pdf_files and st.button("開始總結", key="summarize_button"):
    with st.spinner("正在讀取 PDF 內容並進行 OCR 辨識..."):
        full_document_text = get_pdf_content_with_ocr(pdf_files)
    # 顯示提取出的文字（方便除錯）
    with st.expander("點此查看所有提取出的文字內容"):
        st.text_area("所有文件文字內容", value=full_document_text, height=500)
        
    if full_document_text.strip():
        with st.spinner("正在使用 LLM 總結文件..."):
            summary = get_ollama_summary(full_document_text)
        st.success("總結完成！")
        st.subheader("文件總結")
        # st.write(summary)
        # 簡體轉繁體中文
        summary = enforce_traditional(summary)  # 強制轉繁
        st.write(summary)
    else:
        st.warning("沒有可總結的文字，請確保 PDF 檔案內容可被提取或辨識。")

