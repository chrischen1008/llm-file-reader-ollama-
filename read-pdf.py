import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- 安裝必要的函式庫 ---
# 為了避免版本衝突，建議重新安裝或更新。
# pip install --upgrade langchain langchain-community gradio pypdf transformers accelerate bitsandbytes torch langchain-huggingface
# 另外，請確保已安裝 sentence-transformers，以便某些 LangChain 功能能正常運作。
# pip install sentence-transformers

# --- 設定模型 ---
# 使用 4-bit 量化設定來減少 VRAM 使用量，以適應 RTX 3050 Ti
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 載入 tokenizer 和模型
# 已更換為更適合 4GB VRAM 的模型：microsoft/Phi-3-mini-4k-instruct
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

# 使用 HuggingFace Pipeline 建立摘要模型
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    # 這裡可以根據需要調整
)
llm = HuggingFacePipeline(pipeline=pipe)


def summarize_pdf(pdf_file, custom_prompt=""):
    """
    接收一個 PDF 檔案物件，並使用 LangChain 和本地模型進行摘要。
    
    Args:
        pdf_file: Gradio File 元件提供的檔案物件。
        custom_prompt: 一個可選的自定義提示。
        
    Returns:
        生成的 PDF 摘要。
    """
    # 檢查檔案是否被上傳
    if not pdf_file:
        return "請上傳一個 PDF 檔案。"
        
    try:
        # 從檔案物件中取得路徑
        pdf_file_path = pdf_file.name
        
        # 載入 PDF 文件
        loader = PyPDFLoader(pdf_file_path)
        docs = loader.load_and_split()
        
        # 載入摘要鏈
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # 執行摘要，已更新為 `invoke` 方法
        summary = chain.invoke({"input_documents": docs})
        
        # `invoke` 返回一個字典，摘要結果在 'output_text' 鍵中
        return summary.get('output_text', '無法生成摘要。')

    except Exception as e:
        return f"發生錯誤: {e}"

def main():
    """
    設定並啟動 Gradio 應用程式。
    """
    # 使用 gr.File 元件讓使用者上傳檔案
    input_pdf = gr.File(label="在此上傳您的 PDF 檔案", file_types=[".pdf"])
    
    # 摘要輸出欄位
    output_summary = gr.Textbox(label="摘要")
    
    interface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf,
        outputs=output_summary,
        title="PDF 摘要器 (離線版本)",
        description="此應用程式讓您能夠離線摘要您的 PDF 檔案。",
    )
    
    # 在本機運行 Gradio 應用程式
    interface.launch()

# 確保在程式碼執行時呼叫 main 函數
if __name__ == "__main__":
    main()
