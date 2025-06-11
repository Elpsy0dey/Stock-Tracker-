"""
API Configuration for AI Service
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_CONFIG = {
    'API_KEY': os.getenv('OPENAI_API_KEY', ''),
    'API_URL': 'https://free.v36.cm/v1/chat/completions',  # Hardcoded correct URL
    'MODEL': 'gpt-4o-mini',  # Updated to a working model
    'MAX_TOKENS': 1000,
    'TEMPERATURE': 0.7,
    'REQUEST_TIMEOUT': 30,  # seconds
}

# API Settings
TEMPERATURE = 0.7
TOP_P = 0.9
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

# Cache settings
CACHE_ENABLED = True
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Request settings
MAX_RETRIES = 3 