import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from model.jias_chromadb import (
        embeddings_model, tokenizer, model, reranker,
        arabic_collection, english_collection
    )
except ImportError:
    # Fallback for when running as a module from src root
    from jias_chromadb import (
        embeddings_model, tokenizer, model, reranker,
        arabic_collection, english_collection
    )

def detect_language(text):
    """Detect if text is Arabic or English based on character set"""
    if re.search(r'[\u0600-\u06FF]', text):
        return "arabic"
    return "english"

def search_chromadb(query, top_k=10, lang="english"):
    """Search ChromaDB collection based on language"""
    query_embedding = embeddings_model.embed_query(query)
    
    # Select the appropriate collection
    collection = arabic_collection if lang == "arabic" else english_collection
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    return results

def build_prompt(query, context_texts, lang="english"):
    context_str = "\n".join(context_texts)
    
    if lang == "arabic":
        system_instruction = """أنت مساعد ذكي ومتخصص في دعم أولياء أمور الأطفال المصابين بالتوحد.
لديك معرفة عميقة بطيف التوحد والتحديات التي تواجه الأطفال وأسرهم.
دورك هو تقديم إجابات واضحة ومتعاطفة ومفيدة بناءً على أفضل الممارسات والأبحاث العلمية.

⚠️ إرشادات السلامة الهامة:
- **لا تقدم وصفات طبية:** أنت لست طبيباً. لا توصي بالأدوية أو علاجات طبية محددة أو تشخيصات.
- **معلومات تعليمية فقط:** قدم إرشادات تربوية ونصائح عملية ومعلومات مبنية على الأدلة.
- **الإحالة للمتخصصين:** في حالات القلق الطبي الجاد أو الأعراض الخطيرة، انصح الأهل باستشارة طبيب مختص أو معالج سلوكي.

📋 مبادئ الإجابة:
- استخدم السياق المقدم أدناه كمصدر أساسي للمعلومات
- أجب باللغة العربية الفصحى الواضحة والبسيطة
- كن متعاطفاً ومشجعاً - تذكر أن الأهل قد يشعرون بالقلق أو الإحباط
- قدم نصائح عملية قابلة للتطبيق في الحياة اليومية
- رتب إجابتك بشكل منظم (نقاط أو خطوات) عندما يكون ذلك مناسباً
- إذا كان السؤال خارج نطاق التوحد، وجه المستخدم بلطف للموضوع الأساسي
"""
    else:
        system_instruction = """You are a knowledgeable and specialized assistant supporting parents of children with autism.
You have deep expertise in autism spectrum disorder and the challenges faced by children and their families.
Your role is to provide clear, empathetic, and helpful answers based on best practices and scientific research.

⚠️ IMPORTANT SAFETY GUIDELINES:
- **NO MEDICAL PRESCRIPTIONS:** You are NOT a doctor. DO NOT recommend medications, specific medical treatments, or diagnoses.
- **EDUCATIONAL INFORMATION ONLY:** Provide educational guidance, practical advice, and evidence-based information.
- **REFER TO PROFESSIONALS:** For serious medical concerns, severe symptoms, or emergencies, advise parents to consult a qualified doctor or behavioral therapist.

📋 RESPONSE PRINCIPLES:
- Use the provided context below as your primary source of information
- Respond in clear, simple English
- Be empathetic and encouraging - remember parents may feel anxious or frustrated
- Provide practical, actionable advice for daily life
- Organize your answer clearly (use bullet points or steps when appropriate)
- If the question is outside the scope of autism support, gently guide the user back to the main topic
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
    print(f"Detected language: {lang}")
    
    # Search in the specific language collection
    results = search_chromadb(query, top_k=15, lang=lang)
    
    # Extract passages from results
    passages = results['documents'][0] if results['documents'] else []
    
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
