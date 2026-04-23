# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from langchain_huggingface import HuggingFacePipeline
# from src.config import LLM_MODEL_ID, USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE

# def get_llm_pipeline():
#     print(f"Loading model: {LLM_MODEL_ID}")
    
#     quantization_config = None
#     if USE_4BIT:
#         compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
#             bnb_4bit_compute_dtype=compute_dtype,
#             bnb_4bit_use_double_quant=True,
#         )

#     tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         LLM_MODEL_ID,
#         quantization_config=quantization_config,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.1,
#         return_full_text=False 
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm, tokenizer

# class Generator:
#     def __init__(self):
#         self.llm, self.tokenizer = get_llm_pipeline()

#     def generate_response(self, query: str, context: list):
#         # Format context
#         context_str = "\n\n".join([doc.page_content for doc in context])
        
#         # System prompt: Strict Egyptian Arabic & Context adherence
#         system_prompt = (
#             "You are a helpful, smart assistant. You speak fluent Egyptian Arabic (Masri). "
#             "You answer accurately based ONLY on the provided context. "
#             "1. Do NOT make up information. If the answer is not in the context, say 'أنا آسف، المعلومة دي مش عندي' (I'm sorry, I don't have this info). "
#             "2. Do NOT list sources or references at the end of your answer. "
#             "3. Keep your answer concise and friendly."
#         )
        
#         # Enforcing specific format: Context -> User -> Answer
#         user_prompt = f"Context: {context_str} \n User: {query} \n Answer:"
        
#         # Construct messages
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
        
#         # Apply chat template
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
#         # Generate
#         response = self.llm.invoke(prompt)
#         return response

# if __name__ == "__main__":
#     gen = Generator()
#     response = gen.generate_response("ما هو علاج التوحد؟", [])
#     print(response)



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from src.config import LLM_MODEL_ID, USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE

def get_llm_pipeline():
    print(f"Loading model: {LLM_MODEL_ID}")
    
    quantization_config = None
    if USE_4BIT:
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # Increased to prevent cutoff
        do_sample=True,
        temperature=0.4, # Lowered slightly for stability
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False 
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer


class Generator:
    def __init__(self):
        self.llm, self.tokenizer = get_llm_pipeline()

    def generate_response(self, query: str, context: list):
        # Format context
        context_str = "\n\n".join([doc.page_content for doc in context])
        
        # 1. Detect Language (Simple Regex for Arabic)
        import re
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        
        # 2. Select System Prompt based on Language
        if is_arabic:
            system_prompt = (
                "أنت 'اوتيميت'، مساعد ذكي متخصص في التوحد. "
                "تتحدث باللهجة المصرية (المصري) بطلاقة. "
                "1. جاوب بناءً على السياق المقدم فقط. "
                "2. ممنوع اختلاق معلومات. لو المعلومة مش موجودة، قول 'أنا آسف، المعلومة دي مش عندي'. "
                "3. ممنوع استخدام اللغة الصينية نهائياً. "
                "4. لا تذكر المصادر في نهاية الإجابة."
            )
        else:
            system_prompt = (
                "You are Autimate, a specialized assistant for autism support. "
                "1. Answer in CLEAR ENGLISH. "
                "2. Answer based ONLY on the provided context. "
                "3. Do NOT make up info. If not in context, say 'I don't know'. "
                "4. Do NOT use Chinese or Arabic characters. "
                "5. Do NOT list sources/references."
            )
        
        # Enforcing specific format: Context -> User -> Answer
        user_prompt = f"Context: {context_str} \n User: {query} \n Answer:"
        
        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        response = self.llm.invoke(prompt)
        
        # Post-process cleanup
        import re
        # Remove Bullet points artifacts like \uf076, \uf0a7
        response = re.sub(r'[\uf000-\uf8ff]', '', response)
        # Fix mixed newlines
        response = response.replace("\\n", "\n").strip()
        
        return response

if __name__ == "__main__":
    gen = Generator()
    response = gen.generate_response("ما هو علاج التوحد؟", [])
    print(response)


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from langchain_huggingface import HuggingFacePipeline
# from src.config import LLM_MODEL_ID, USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE

# def get_llm_pipeline():
#     print(f"Loading model: {LLM_MODEL_ID}")
    
#     quantization_config = None
#     if USE_4BIT:
#         compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
#             bnb_4bit_compute_dtype=compute_dtype,
#             bnb_4bit_use_double_quant=True,
#         )

#     tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         LLM_MODEL_ID,
#         quantization_config=quantization_config,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=1.1,
#         return_full_text=False 
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm, tokenizer

# class Generator:
#     def __init__(self):
#         self.llm, self.tokenizer = get_llm_pipeline()

#     def generate_response(self, query: str, context: list):
#         # Format context
#         context_str = "\n\n".join([doc.page_content for doc in context])
        
#         # System prompt optimized for Arabic/MSA/Egyptian
#         system_prompt = """أنت "BrainLight"، مساعد ذكي متخصص في التوحد، تتحدث باللهجة المصرية والعربية الفصحى.
#         مهمتك هي مساعدة المستخدمين بناءً على السياق المقدم فقط.
#         إذا لم تجد الإجابة في السياق، قل بوضوح أنك لا تعرف.
#         """
        
#         # Construct messages for Qwen chat template
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"السياق:\n{context_str}\n\nالسؤال: {query}"}
#         ]
        
#         # Apply chat template
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
#         # Generate
#         response = self.llm.invoke(prompt)
#         return response

# if __name__ == "__main__":
#     gen = Generator()
#     response = gen.generate_response("ما هو علاج التوحد؟", [])
#     print(response)
