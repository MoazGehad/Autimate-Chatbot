import sys
import os
from pathlib import Path

# Ensure the root directory (chatbot_to_upload) is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from src.llm_provider import BaseLLMProvider
from src.logging_config import get_logger, log_latency
from langchain_core.documents import Document

logger = get_logger(__name__)

class Generator:
    """Thin facade around the active LLM provider."""
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def generate_response(self, query: str, context: List[Document]) -> str:
        # Format context
        context_str = "\n\n".join([doc.page_content for doc in context])
        
        logger.info(f"Generating response for query length {len(query)} with {len(context)} docs context")
        
        with log_latency(logger, "generate_response"):
            response = await self.provider.generate(query, context_str)
            
        return response
