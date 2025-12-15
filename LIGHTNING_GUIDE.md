# Lightning.ai Deployment Guide

## 1. Environment Variables
Ensure your `.env` file (or Lightning.ai Secrets) contains the following keys:

```bash
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
# HF_TOKEN is not strictly required for Qwen/Qwen2.5-7B-Instruct as it is open, 
# but good to have if you switch to gated models.
# HF_TOKEN=your_huggingface_token 
```

## 2. Port Exposure (Public URL)
To allow your Flutter app to access the API running on Lightning.ai:

1.  **Start the API**: Run the API server in the terminal.
    ```bash
    python src/model/api.py
    ```
    You should see output indicating it is running on `0.0.0.0:8000`.

2.  **Expose the Port**:
    *   Look at the **Ports** tab or the "Open Ports" button in the Lightning Studio interface.
    *   Add port `8000`.
    *   It will generate a **Public URL** (e.g., `https://your-studio-name-8000.lightning.ai`).

3.  **Update Flutter App**:
    *   Copy this Public URL.
    *   Paste it into your `ChatService` in `src/flutter/chat_service.dart` (or use the `updateBaseUrl` method).

## 3. Troubleshooting
*   **OOM Errors**: If you see "Out of Memory", ensure you are using a GPU Studio (T4 or A10G). The code is configured for 4-bit quantization to fit in ~16GB VRAM.
*   **Timeout**: If the first request times out, it might be because the model is loading. Wait a minute and try again.
*   **CORS**: The API is configured to allow all origins (`*`), so your mobile app should not face CORS issues.

## 4. Testing on Lightning.ai (Before Flutter)

### Option A: Interactive Terminal Chat
You can chat with the model directly in the Lightning Studio terminal without starting the API.
1.  **Run the script**:
    ```bash
    python src/model/chatbot.py
    ```
2.  **Chat**: Type your question when prompted (e.g., "What are early signs of autism?").
3.  **Exit**: Type `exit` or `quit` to stop.

### Option B: Testing the API
If you want to test the server (`api.py`) to ensure it handles requests correctly before connecting the app:

1.  **Start the API** (if not already running):
    ```bash
    python src/model/api.py
    ```
2.  **Open a New Terminal**: Click the `+` icon to open a second terminal tab in Lightning Studio.
3.  **Send a Test Request**: Run the following command:
    ```bash
    curl -X POST "http://127.0.0.1:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"question": "How can I help my autistic child sleep better?"}'
    ```
4.  **Check Response**: You should see a JSON response with the answer (e.g., `{"answer": "..."}`).

## 5. Optimizing Upload Size
To reduce the upload size and speed up deployment, you can **exclude** the following folders. The chatbot only needs the code and the connection to Pinecone; it does not need the raw data files.

**Folders to Exclude:**
*   `.venv` (Lightning creates its own environment)
*   `.idea` or `.vscode` (Editor settings)
*   `datasets` (Raw data is already in Pinecone)
*   `processed` (Processed data is already in Pinecone)
*   `__pycache__` (Python cache files)
*   `.git` (Version control history, optional)

**Files to Keep:**
*   `src/` (All source code)
*   `requirements.txt`
*   `.env` (Or set these as Secrets in Lightning Studio)
*   `LIGHTNING_GUIDE.md`


