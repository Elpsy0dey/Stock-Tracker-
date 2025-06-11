"""
Model Manager Service

Handles checking which AI models are available and selecting the best one.
Runs automatically on application startup.
"""

import os
import requests
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from config.api_config import API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

# List of models to check in priority order (most preferred first)
PREFERRED_MODELS = [
    "gpt-4o-mini",          # Best quality
    "gpt-3.5-turbo-16k",    # Large context window
    "gpt-3.5-turbo-0125",   # Good quality
    "gpt-3.5-turbo-1106",   # Good fallback
    "gpt-3.5-turbo"         # Original model
]

class ModelManager:
    """Manages AI model selection based on availability"""
    
    @classmethod
    def initialize(cls) -> str:
        """
        Run on application startup to check available models and select the best one.
        
        Returns:
            str: The selected model name
        """
        logger.info("Initializing Model Manager")
        selected_model = cls.check_and_select_best_model()
        
        if selected_model:
            logger.info(f"Selected model: {selected_model}")
            
            # Update the API_CONFIG globally
            API_CONFIG['MODEL'] = selected_model
            
            return selected_model
        else:
            logger.warning("No working models found! Using default model.")
            return API_CONFIG['MODEL']
    
    @classmethod
    def check_and_select_best_model(cls) -> Optional[str]:
        """
        Check which models are available and select the best one based on preference order.
        
        Returns:
            Optional[str]: The best available model or None if none are working
        """
        # Check if API key and URL are configured
        if not API_CONFIG['API_KEY'] or not API_CONFIG['API_URL']:
            logger.warning("API key or URL not configured. Skipping model check.")
            return None
        
        available_models = []
        
        for model in PREFERRED_MODELS:
            if cls.test_model(model):
                available_models.append(model)
                logger.info(f"Model {model} is working")
                
                # Return the first working model (already in preference order)
                return model
            else:
                logger.info(f"Model {model} is not available")
            
            # Add a small delay to prevent rate limiting
            time.sleep(1)
            
        return None
    
    @classmethod
    def test_model(cls, model_name: str) -> bool:
        """
        Test if a specific model is working with the API
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            bool: True if model is working, False otherwise
        """
        logger.info(f"Testing model availability: {model_name}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['API_KEY']}"
        }
        
        # Minimal request payload
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        try:
            # Make a request to the API
            response = requests.post(
                API_CONFIG['API_URL'],
                headers=headers,
                json=payload,
                timeout=10  # Short timeout for checking
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Check if the response has the expected structure
                if 'choices' in result and result['choices'] and 'message' in result['choices'][0]:
                    # Check for unauthorized messages in the content
                    content = result['choices'][0]['message'].get('content', '').lower()
                    
                    if 'unauthorized' not in content and 'auth' not in content:
                        logger.info(f"✅ Model {model_name} is working!")
                        return True
            
            logger.info(f"❌ Model {model_name} failed check: {response.status_code}")
            return False
            
        except Exception as e:
            logger.warning(f"Error testing model {model_name}: {str(e)}")
            return False 