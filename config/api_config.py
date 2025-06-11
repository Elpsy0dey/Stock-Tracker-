"""
API Configuration for AI Service

Loads configuration from environment variables for secure deployment
"""
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_config")

# Load environment variables from .env file (if it exists)
# In production, these would be set in the environment directly
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

# API Configuration
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
    logger.warning("No API key found! Set OPENAI_API_KEY in environment or .env file")

# API Settings
TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
TOP_P = float(os.getenv('OPENAI_TOP_P', '0.9'))
FREQUENCY_PENALTY = float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0'))
PRESENCE_PENALTY = float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0'))

# Cache settings
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', '3600'))  # 1 hour in seconds

# Request settings
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3')) 