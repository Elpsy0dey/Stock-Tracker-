"""
API Configuration for AI Service

Loads configuration from Streamlit secrets or environment variables for secure deployment
"""
import os
import logging
import streamlit as st  # Add Streamlit import
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_config")

# Load environment variables from .env file (if it exists)
# In production on Streamlit Cloud, these would be set in Streamlit Secrets
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    logger.info("Loading configuration from .env file")
    load_dotenv(dotenv_path)
else:
    logger.info("No .env file found, using environment variables")
    load_dotenv()  # Will still try to load from default location

# Default API URL if not specified in environment variables
DEFAULT_API_URL = 'https://free.v36.cm/v1/chat/completions'
DEFAULT_MODEL = 'gpt-4o-mini'

# API Configuration - prioritize Streamlit secrets if available
try:
    # Check if running in Streamlit
    if 'secrets' in dir(st):
        # Use Streamlit secrets
        API_CONFIG = {
            'API_KEY': st.secrets.get("OPENAI_API_KEY", ""),
            'API_URL': st.secrets.get("OPENAI_API_URL", DEFAULT_API_URL),
            'MODEL': st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL),
            'MAX_TOKENS': int(st.secrets.get("OPENAI_MAX_TOKENS", "1000")),
            'TEMPERATURE': float(st.secrets.get("OPENAI_TEMPERATURE", "0.7")),
            'REQUEST_TIMEOUT': int(st.secrets.get("OPENAI_REQUEST_TIMEOUT", "30")),
        }
        logger.info("Using Streamlit secrets for configuration")
    else:
        # Fall back to environment variables
        API_CONFIG = {
            'API_KEY': os.getenv('OPENAI_API_KEY', ''),
            'API_URL': os.getenv('OPENAI_API_URL', DEFAULT_API_URL),
            'MODEL': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
            'MAX_TOKENS': int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
            'TEMPERATURE': float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            'REQUEST_TIMEOUT': int(os.getenv('OPENAI_REQUEST_TIMEOUT', '30')),
        }
        logger.info("Using environment variables for configuration")
except Exception as e:
    # Fall back to environment variables if Streamlit isn't available
    logger.warning(f"Error using Streamlit secrets: {e}. Falling back to environment variables.")
    API_CONFIG = {
        'API_KEY': os.getenv('OPENAI_API_KEY', ''),
        'API_URL': os.getenv('OPENAI_API_URL', DEFAULT_API_URL),
        'MODEL': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'MAX_TOKENS': int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
        'TEMPERATURE': float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
        'REQUEST_TIMEOUT': int(os.getenv('OPENAI_REQUEST_TIMEOUT', '30')),
    }

# Check if API key is set
if not API_CONFIG['API_KEY']:
    logger.warning("No API key found! Set OPENAI_API_KEY in environment, .env file, or Streamlit secrets")

# API Settings - use the same logic
try:
    if 'secrets' in dir(st):
        TEMPERATURE = float(st.secrets.get("OPENAI_TEMPERATURE", "0.7"))
        TOP_P = float(st.secrets.get("OPENAI_TOP_P", "0.9"))
        FREQUENCY_PENALTY = float(st.secrets.get("OPENAI_FREQUENCY_PENALTY", "0.0"))
        PRESENCE_PENALTY = float(st.secrets.get("OPENAI_PRESENCE_PENALTY", "0.0"))
    else:
        TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        TOP_P = float(os.getenv('OPENAI_TOP_P', '0.9'))
        FREQUENCY_PENALTY = float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0'))
        PRESENCE_PENALTY = float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0'))
except Exception:
    TEMPERATURE = 0.7
    TOP_P = 0.9
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0

# Cache settings
try:
    if 'secrets' in dir(st):
        CACHE_ENABLED = st.secrets.get("CACHE_ENABLED", "True").lower() == 'true'
        CACHE_EXPIRY = int(st.secrets.get("CACHE_EXPIRY", "3600"))
    else:
        CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
        CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', '3600'))
except Exception:
    CACHE_ENABLED = True
    CACHE_EXPIRY = 3600

# Request settings
try:
    if 'secrets' in dir(st):
        MAX_RETRIES = int(st.secrets.get("MAX_RETRIES", "3"))
    else:
        MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
except Exception:
    MAX_RETRIES = 3 