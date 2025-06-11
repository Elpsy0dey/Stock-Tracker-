"""
Startup Utilities

This module handles various tasks that should run on application startup
"""

import logging
from services.model_manager import ModelManager
from config.api_config import API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("startup")

def initialize_application():
    """
    Initialize all application components that need to run on startup.
    
    This function handles:
    1. Checking and selecting the best AI model
    2. Other initialization tasks as needed
    """
    logger.info("Starting application initialization")
    
    # Step 1: Initialize AI model selection
    initialize_ai_models()
    
    # Add other initialization steps here as needed
    
    logger.info("Application initialization complete")

def initialize_ai_models():
    """
    Check available AI models and select the best one.
    """
    logger.info("Initializing AI model selection")
    
    try:
        # Use the ModelManager to check and select the best model
        selected_model = ModelManager.initialize()
        
        logger.info(f"AI model selection complete. Using model: {selected_model}")
        
        # Return the selected model (useful for debugging or reporting)
        return selected_model
    
    except Exception as e:
        logger.error(f"Error initializing AI models: {str(e)}")
        logger.warning("Using default model as fallback")
        return API_CONFIG['MODEL'] 