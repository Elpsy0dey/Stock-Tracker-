"""
Test script to verify that the API configuration correctly loads from Streamlit secrets
"""

import streamlit as st
from config.api_config import API_CONFIG

def main():
    st.set_page_config(page_title="API Config Test", page_icon="🔐")
    
    st.title("🔐 API Configuration Test")
    st.write("This app tests whether your API configuration is correctly loaded.")

    # Show current configuration (with masked API key)
    st.header("Current API Configuration")
    
    # Mask API key for security
    api_key = API_CONFIG['API_KEY']
    masked_key = "••••••" + api_key[-4:] if len(api_key) > 4 else "••••••"
    
    st.json({
        "API_KEY": masked_key,
        "API_URL": API_CONFIG['API_URL'],
        "MODEL": API_CONFIG['MODEL'],
        "MAX_TOKENS": API_CONFIG['MAX_TOKENS'],
        "TEMPERATURE": API_CONFIG['TEMPERATURE'],
        "REQUEST_TIMEOUT": API_CONFIG['REQUEST_TIMEOUT']
    })
    
    # Show where configuration was loaded from
    st.header("Configuration Source")
    
    # Check if using Streamlit secrets
    if 'secrets' in dir(st) and 'OPENAI_API_KEY' in st.secrets:
        st.success("✅ Configuration loaded from **Streamlit Secrets**")
        
        # Check if keys match what's in secrets (without revealing the actual keys)
        if st.secrets.get("OPENAI_API_KEY", "")[-4:] == api_key[-4:]:
            st.success("✅ API key matches Streamlit secrets")
        else:
            st.error("❌ API key does not match Streamlit secrets")
            
    # Check if environment variables are being used
    elif api_key:
        st.info("ℹ️ Configuration loaded from **Environment Variables**")
    else:
        st.warning("⚠️ No API key found. Please check your configuration.")
    
    # Show instructions
    st.header("Setup Instructions")
    
    st.markdown("""
    ### Local Development
    
    1. Create a `.streamlit/secrets.toml` file with:
    ```toml
    OPENAI_API_KEY = "your-api-key"
    OPENAI_API_URL = "https://free.v36.cm/v1/chat/completions"
    OPENAI_MODEL = "gpt-4o-mini"
    ```
    
    ### Streamlit Cloud Deployment
    
    1. Go to your app's settings
    2. Add the same TOML content in the Secrets section
    3. Save changes
    """)

if __name__ == "__main__":
    main() 