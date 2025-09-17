import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- 安裝必要的函式庫 ---
# pip install --upgrade langchain langchain-community gradio pypdf transformers accelerate bitsandbytes torch langchain-huggingface
# 另外，請確保已安裝 sentence-transformers
# pip install sentence-transformers

# --- 設定模型 ---
# 針對 12GB VRAM，選擇 Llama 3.1 8B Instruct 並使用 4-bit 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 選擇 Llama 3.1 8B Instruct 模型
# 請注意，首次使用此模型可能需要您在 Hugging Face 網站上接受其使用條款。
model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
llm = HuggingFacePipeline(pipeline=pipe)


def summarize_pdf(pdf_file):
    """
    接收一個 PDF 檔案物件，並使用 LangChain 和本地模型進行摘要。
    
    Args:
    pdf_file: Gradio File 元件提供的檔案物件。
    
    Returns:
    生成的 PDF 摘要。
    """
    if not pdf_file:
        return "請上傳一個 PDF 檔案。"
        
    try:
        pdf_file_path = pdf_file.name
        
        loader = PyPDFLoader(pdf_file_path)
        docs = loader.load_and_split()
        
        # --- 變更開始 ---
        # 建立一個明確要求以繁體中文回應的提示模板
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="您是一個專業的摘要專家。請根據提供的文件內容，以繁體中文為我生成一份精簡且準確的摘要。"
                ),
                HumanMessage(
                    content="文件內容:\n{text}"
                ),
            ]
        )

        # 使用 `stuff` 鏈，適合短文件，更高效。
        # 現在我們將自定義的中文提示模板傳入，以確保回應是中文。
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
        # --- 變更結束 ---
        
        summary = chain.invoke({"input_documents": docs})
        
        return summary.get('output_text', '無法生成摘要。')

    except Exception as e:
        return f"發生錯誤: {e}"

def main():
    """
    設定並啟動 Gradio 應用程式。
    """
    input_pdf = gr.File(label="在此上傳您的 PDF 檔案", file_types=[".pdf"])
    output_summary = gr.Textbox(label="摘要")
    
    interface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf,
        outputs=output_summary,
        title="PDF 摘要器 (Llama 3.1 8B 版本)",
        description="此應用程式使用強大的 Llama 3.1 8B 模型，為您離線摘要 PDF 檔案。",
    )
    
    interface.launch()

if __name__ == "__main__":
    main()

