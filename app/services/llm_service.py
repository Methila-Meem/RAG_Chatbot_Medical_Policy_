from groq import Groq
from app.config import get_settings
from typing import List

settings = get_settings()

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
    
    def generate_answer(self, question: str, context: str, conversation_history: List[dict] = None) -> str:
        """Generate answer using Groq"""
        
        system_prompt = """You are a helpful medical policy assistant. 
        Answer questions based ONLY on the provided context. 
        If the answer is not in the context, say "I don't have enough information to answer this question."
        Be concise and professional. Cite the source when possible."""
        
        user_prompt = f"""Context:
        {context}

        Question: {question}

        Answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"