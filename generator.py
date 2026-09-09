import re
import os
import sys
from typing import Dict, Any, Optional

# Since we are in the judge folder, we can import directly
try:
    from LLM import LLM
except ImportError:
    # Fallback
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from LLM import LLM

def extract_html_content(text: str) -> str:
    """Extract HTML content from model response"""
    # 1. Look for code blocks
    code_block_pattern = r"```html\s*([\s\S]*?)```"
    match = re.search(code_block_pattern, text)
    if match:
        return match.group(1).strip()
    
    # 2. Look for doctype or html tag
    if "<!DOCTYPE html" in text or "<html" in text:
        # Try to find the start and end
        start = text.find("<!DOCTYPE html")
        if start == -1:
            start = text.find("<html")
        
        end = text.rfind("</html>")
        if start != -1 and end != -1:
            return text[start:end+7].strip()
        if start != -1:
            return text[start:].strip()
            
    return text.strip()

class HTMLGenerator:
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the generator with model configuration.
        model_config: {
            "model_name": str,
            "api_key": str,
            "base_url": str,
            "is_local": bool
        }
        """
        self.config = model_config
        self.llm = LLM(
            model_name=model_config.get("model_name", "gpt-3.5-turbo"),
            api_key=model_config.get("api_key", "EMPTY"),
            base_url=model_config.get("base_url", None)
        )

    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate HTML code based on the prompt.
        """
        default_system_prompt = "You are a helpful expert web developer. Please generate a single-file HTML5 application based on the user's request. Ensure the code is complete, functional, and self-contained (CSS and JS included). Do not use external CDNs if possible, or use reliable ones."
        
        final_system_prompt = system_prompt if system_prompt and system_prompt.strip() else default_system_prompt
        
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Pass generation parameters to LLM
            response = self.llm(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            content = response['content'] if isinstance(response, dict) else response
            return extract_html_content(content)
        except Exception as e:
            print(f"Generation failed: {e}")
            return ""
