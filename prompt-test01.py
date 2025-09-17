def get_ollama_summary(text):
    """
    將提取的文字傳送給 Ollama 模型進行總結（強制繁體中文）。
    """
    if not text.strip():
        return "沒有可總結的文字。"

    # 強化語言要求，多次重複
    language_instruction = (
        "你是一位專業的文件分析助理，必須全程使用繁體中文回覆。" 
        "禁止使用任何英文或簡體中文。輸出必須完全以繁體中文呈現。" 
        "即使文件內容包含英文，也必須將它們翻譯成繁體中文再輸出。" 
        "請務必全程使用繁體中文回答，請務必全程使用繁體中文回答，請務必全程使用繁體中文回答。"
    )

    # 將文件內容包裝成強化 prompt
    prompt = (
        f"{language_instruction}\n\n"
        f"請幫我詳細總結以下文件內容，條列出所有重點，包含圖片 OCR 辨識到的文字內容：\n\n"
        f"{text[:20000]}"
    )

    try:
        response = ollama.chat(
            #model='gemma3:12b',
            model='qwen2:7b',#偶爾會出現簡體中文
            messages=[
                {'role': 'system', 'content': language_instruction},
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"與 Ollama 溝通時發生錯誤。請確認 Ollama 服務已啟動且模型 'gemma3:latest' 已下載。錯誤訊息：{e}"