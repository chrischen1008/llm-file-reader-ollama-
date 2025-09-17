def get_ollama_summary(text):
    """
    將提取的文字傳送給 Ollama 模型進行總結（強制繁體中文）。
    """
    if not text.strip():
        return "沒有可總結的文字。"

    system_prompt = """
                    ###你只能用繁體中文輸出!###
                    根據使用者所提供的內文依以下規則完成整理重點：
                    - 將內容濃縮成清晰的條列重點
                    - 保留關鍵專有名詞與重要數字
                    - 全文使用繁體中文
                    """

    # 將文件內容包裝成強化 prompt
    prompt = (
        f"{system_prompt}\n\n"
        f"""請閱讀我上傳的 PDF，並依以下規則整理重點：
            1. 摘要目標：
               - 將內容濃縮成清晰的條列重點               

            2. 摘要規則：
               - 每頁整理 3~5 個重點
               - 每個重點不超過 25 字
               - 保留關鍵專有名詞與重要數字
               - [選填] 標註原始 PDF 頁碼

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