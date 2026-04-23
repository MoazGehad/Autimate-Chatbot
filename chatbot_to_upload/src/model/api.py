from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import ask_chatbot 
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile app access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Model Loaded and API is ready"}

@app.post("/chat")
def chat_endpoint(query: Query):
    full_text = ask_chatbot(query.question)

    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[-1].strip()
    else:
        answer = full_text.strip()

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
