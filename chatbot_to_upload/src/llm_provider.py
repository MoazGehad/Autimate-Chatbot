from abc import ABC, abstractmethod
import asyncio
import re
from typing import List
from src.logging_config import get_logger

logger = get_logger(__name__)

def detect_language_and_get_prompt(query: str) -> tuple[bool, str]:
    """Detects if query is Arabic and returns a tuple of (is_arabic, system_prompt)."""
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    
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
    return is_arabic, system_prompt

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: str) -> str:
        pass
        
    @abstractmethod
    async def generate_query_rewrite(self, query: str) -> List[str]:
        pass

class QwenProvider(BaseLLMProvider):
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        from langchain_huggingface import HuggingFacePipeline
        from src.config import LLM_MODEL_ID, USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE
        
        logger.info(f"Loading local model: {LLM_MODEL_ID}")
        
        quantization_config = None
        if USE_4BIT:
            compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False 
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Local Qwen model loaded successfully")

    async def generate(self, query: str, context: str) -> str:
        is_arabic, system_prompt = detect_language_and_get_prompt(query)
        user_prompt = f"Context: {context} \n User: {query} \n Answer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.llm.invoke, prompt)
        
        # Post-process cleanup
        response = re.sub(r'[\uf000-\uf8ff]', '', response)
        response = response.replace("\\n", "\n").strip()
        
        return response

    async def generate_query_rewrite(self, query: str) -> List[str]:
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        
        if is_arabic:
            system_prompt = "أنت مساعد ذكي. مهمتك هي إعادة صياغة سؤال المستخدم إلى 2-3 أسئلة قصيرة ومختلفة المعنى لتحسين البحث في قاعدة البيانات. أعد الأسئلة كقائمة مفصولة بفواصل دون أي نص إضافي."
        else:
            system_prompt = "You are an intelligent assistant. Rewrite the user's query into 2-3 short, semantically distinct sub-queries to improve database search. Return ONLY a comma-separated list of queries."
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.llm.invoke, prompt)
        
        # Parse comma-separated list
        queries = [q.strip() for q in response.split(',') if q.strip()]
        return queries[:3] if queries else [query]


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        import google.generativeai as genai
        from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Gemini provider initialized with model: {GEMINI_MODEL_NAME}")

    async def generate(self, query: str, context: str) -> str:
        is_arabic, system_prompt = detect_language_and_get_prompt(query)
        user_prompt = f"Context: {context} \n User: {query} \n Answer:"
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # In an async context, running synchronous google-generativeai client in executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.model.generate_content(full_prompt))
        
        if not response.text:
            return "I'm sorry, I couldn't generate a response."
            
        text = response.text
        # Cleanup
        text = re.sub(r'[\uf000-\uf8ff]', '', text)
        return text.strip()
        
    async def generate_stream(self, query: str, context: str):
        is_arabic, system_prompt = detect_language_and_get_prompt(query)
        user_prompt = f"Context: {context} \n User: {query} \n Answer:"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                # Cleanup and yield
                text = re.sub(r'[\uf000-\uf8ff]', '', chunk.text)
                yield text

    async def generate_query_rewrite(self, query: str) -> List[str]:
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        
        if is_arabic:
            system_prompt = "أنت مساعد ذكي. مهمتك هي إعادة صياغة سؤال المستخدم إلى 2-3 أسئلة قصيرة ومختلفة المعنى لتحسين البحث في قاعدة البيانات. أعد الأسئلة كقائمة مفصولة بفواصل دون أي نص إضافي."
        else:
            system_prompt = "You are an intelligent assistant. Rewrite the user's query into 2-3 short, semantically distinct sub-queries to improve database search. Return ONLY a comma-separated list of queries."
            
        full_prompt = f"{system_prompt}\n\nQuery: {query}"
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.model.generate_content(full_prompt))
        
        if not response.text:
            return [query]
            
        # Parse comma-separated list
        queries = [q.strip() for q in response.text.split(',') if q.strip()]
        return queries[:3] if queries else [query]

def get_llm_provider(backend: str) -> BaseLLMProvider:
    if backend == "gemini":
        return GeminiProvider()
    elif backend == "local":
        return QwenProvider()
    else:
        logger.warning(f"Unknown backend '{backend}', falling back to local Qwen.")
        return QwenProvider()
