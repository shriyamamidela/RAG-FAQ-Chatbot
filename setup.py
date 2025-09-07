#!/usr/bin/env python3
"""
FAQ Chatbot Setup Script

This script helps you set up the FAQ chatbot quickly.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    # Try minimal requirements first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"])
        print("✅ Minimal requirements installed successfully")
        
        # Try to install advanced requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
            print("✅ Advanced requirements installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️ Advanced requirements failed, but minimal requirements are sufficient")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    try:
        with open(".env", "w") as f:
            f.write("""# Environment Variables for FAQ Chatbot

# Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Application Configuration
APP_TITLE=FAQ Chatbot
APP_DESCRIPTION=Ask questions about Myntra's policies and services
""")
        print("✅ .env file created")
        print("⚠️  Please add your Gemini API key to the .env file")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def check_faq_file():
    """Check if FAQ file exists."""
    if os.path.exists("faqs.txt"):
        print("✅ FAQ file (faqs.txt) found")
        return True
    else:
        print("❌ FAQ file (faqs.txt) not found")
        return False

def main():
    """Main setup function."""
    print("🚀 FAQ Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check FAQ file
    if not check_faq_file():
        print("Please ensure faqs.txt exists in the current directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your Gemini API key to the .env file")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
