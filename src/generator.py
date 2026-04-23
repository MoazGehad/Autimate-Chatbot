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
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False 
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer

class Generator:
    def __init__(self, llm=None, tokenizer=None):
        if llm and tokenizer:
            self.llm = llm
            self.tokenizer = tokenizer
        else:
            self.llm, self.tokenizer = get_llm_pipeline()

    def generate_response(self, query: str, context: list):
        # Format context
        context_str = "\n\n".join([doc.page_content for doc in context])
        
        # System prompt optimized for Arabic/MSA/Egyptian
        system_prompt = """أنت "BrainLight"، مساعد ذكي متخصص في التوحد، تتحدث باللهجة المصرية والعربية الفصحى.
        مهمتك هي مساعدة المستخدمين بناءً على السياق المقدم فقط.
        إذا لم تجد الإجابة في السياق، قل بوضوح أنك لا تعرف.
        """
        
        # Construct messages for Qwen chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"السياق:\n{context_str}\n\nالسؤال: {query}"}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        response = self.llm.invoke(prompt)
        return response

if __name__ == "__main__":
    gen = Generator()
    response = gen.generate_response("ما هو علاج التوحد؟", [])
    print(response)
