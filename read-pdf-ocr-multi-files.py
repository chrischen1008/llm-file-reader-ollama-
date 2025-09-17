import streamlit as st
import ollama
import fitz # PyMuPDF
import io
import pytesseract
from PIL import Image

# --- 函數定義 ---

# 設定 pytesseract 的安裝路徑。請根據您的系統調整此路徑。
# 在 Windows 上，路徑可能像這樣：r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 在 macOS/Linux 上，通常不需要設定，如果遇到問題，請確保 tesseract 已安裝並在 PATH 中。
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    st.warning(f"無法設定 Tesseract-OCR 路徑，請確認其已正確安裝並在系統 PATH 中。錯誤：{e}")


def get_pdf_content_with_ocr(pdf_files):
    """
    從多個 PDF 檔案中提取文字和圖片內容。
    對於圖片，它會使用 Tesseract 進行 OCR 辨識，並將結果加入到文件中。
    """
    full_text = ""
    for pdf_file in pdf_files:
        st.write(f"正在處理檔案：**{pdf_file.name}**...")
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
            st.error(f"處理檔案 **{pdf_file.name}** 時發生錯誤：{e}")
            
    return full_text


@st.cache_data
def get_ollama_summary(text):
    """
    將提取的文字傳送給 Ollama 模型進行總結。
    使用 st.cache_data 暫存結果，避免重複計算。
    """
    if not text.strip():
        return "沒有可總結的文字。"
    
    # 提示詞已經是繁體中文，這會讓模型以繁體中文回應
    # 限制輸入長度，避免 Ollama 模型過載
    prompt = f"請以繁體中文回答。請幫我總結以下多個文件內容，並條列出重點，包含圖片中的文字：\n\n{text[:20000]}"
    
    try:
        # 請確認您已啟動 Ollama 服務並下載 `gemma3:latest` 模型
        response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"與 Ollama 溝通時發生錯誤。請確認 Ollama 服務已啟動且模型 'gemma3:latest' 已下載。錯誤訊息：{e}")
        return "總結失敗，請檢查 Ollama 服務設定。"


# --- Streamlit 介面 ---
st.set_page_config(page_title="Gemma 3 PDF 總結器 (多檔支援)", layout="wide")
st.title("使用 Gemma 3 進行多個 PDF 總結 (包含圖片 OCR)")

st.info("此程式會提取上傳的多個 PDF 中的文字，並對圖片進行 OCR 辨識，然後一併交由 Gemma 3 總結。")

# 上傳多個 PDF 檔案
pdf_files = st.file_uploader(
    "請上傳你的 PDF 檔案", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    # 提取文字與圖片內容
    with st.spinner("正在讀取 PDF 內容並進行 OCR 辨識..."):
        full_document_text = get_pdf_content_with_ocr(pdf_files)
        
    st.success("PDF 內容讀取與 OCR 辨識完成！")
    
    # 顯示提取出的文字 (方便除錯)
    with st.expander("點此查看所有提取出的文字內容"):
        st.text_area("所有文件文字內容", value=full_document_text, height=300)
    
    # 總結文字
    if st.button("開始總結", key="summarize_button"):
        if full_document_text.strip():
            with st.spinner("正在使用 Gemma 3 總結文件..."):
                summary = get_ollama_summary(full_document_text)
                
            st.success("總結完成！")
            
            st.subheader("文件總結")
            st.write(summary)
        else:
            st.warning("沒有可總結的文字，請確保 PDF 檔案內容可被提取或辨識。")
