import sys
import os

# Add current directory to path to allow imports when running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from jias import embeddings_model, tokenizer, model, index, reranker
except ImportError:
    # Fallback for when running as a module from src root
    from model.jias import embeddings_model, tokenizer, model, index, reranker

import re

# ... imports ...

def detect_language(text):
    # Simple check for Arabic characters
    if re.search(r'[\u0600-\u06FF]', text):
        return "arabic"
    return "english"

# to get top answers match
def search_pinecone(query, top_k=10, namespace="english"):
    query_embedding = embeddings_model.embed_query(query) #convert to vector 
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return results

def build_prompt(query, context_texts, lang="english"):
    context_str = "\n".join(context_texts)
    
    if lang == "arabic":
        system_instruction = """أنت مساعد ذكي وداعم لأولياء أمور الأطفال المصابين بالتوحد.
دورك هو مساعدة الوالدين على فهم أطفالهم والتوحد بشكل عام، وتقديم إجابات واضحة ومتعاطفة ودقيقة.

إرشادات السلامة الهامة:
- **لا وصفات طبية:** أنت لست طبيباً. لا توصي بالأدوية أو علاجات طبية محددة أو تشخيصات.
- **لأغراض تعليمية فقط:** قدم فقط التوجيه والنصائح والمعلومات التعليمية.
- **الرجوع للمختصين:** في حالات القلق الطبي الجدي أو الطوارئ، انصح المستخدم صراحة باستشارة طبيب.

إرشادات لك:
- استخدم السياق المقدم للإجابة.
- أجب باللغة العربية فقط.
- كن داعماً ومتعاطفاً.
"""
    else:
        system_instruction = """You are a knowledgeable and supportive assistant for parents of children with autism.  
Your role is to help parents better understand their children, autism in general, and provide clear, empathetic, and accurate answers to their questions.  

IMPORTANT SAFETY GUIDELINES:
- **NO MEDICAL PRESCRIPTIONS:** You are NOT a doctor. DO NOT recommend medications, specific medical treatments, or diagnoses.
- **EDUCATIONAL ONLY:** Provide only guidance, tips, and educational information.
- **REFER TO PROFESSIONALS:** For serious medical concerns, symptoms, or emergencies, explicitly advise the user to consult a doctor or healthcare professional.

Guidelines for you:  
- Always use the provided context to answer.
- Respond in English only.
- Be supportive and empathetic.
"""

    prompt = f"""
{system_instruction}

Context information: 
{context_str}

Question:
{query}

Answer:
"""
    return prompt.strip()

def ask_chatbot(query):
    lang = detect_language(query)
    print(f"Detected language: {lang}") # Debugging
    
    # Search in the specific language namespace
    results = search_pinecone(query, top_k=15, namespace=lang)
    
    matches = results["matches"]
    passages = [match["metadata"].get("text", "") for match in matches]
    
    if passages:
        # Rerank
        pairs = [[query, p] for p in passages]
        scores = reranker.predict(pairs)
        
        # Sort by score
        scored_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        
        # Take top 3
        top_passages = [p for p, s in scored_passages[:3]]
    else:
        top_passages = []

    prompt = build_prompt(query, top_passages, lang=lang)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the new tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Bye! 👋")
            break
        answer = ask_chatbot(query)
        print(f"Chatbot: {answer}")
