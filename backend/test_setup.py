"""
Simple test script to verify the backend setup
Run this to check if all components are working
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
        
        import langchain
        print("✓ LangChain imported successfully")
        
        import chromadb
        print("✓ ChromaDB imported successfully")
        
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("✓ LangChain OpenAI components imported successfully")
        
        from langgraph.prebuilt import create_react_agent
        print("✓ LangGraph imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\nTesting file structure...")
    
    base_dir = Path(__file__).parent.parent
    required_files = [
        base_dir / "mock_data" / "handbooks" / "phishing_response.md",
        base_dir / "mock_data" / "handbooks" / "general_security_policy.md", 
        base_dir / "mock_data" / "handbooks" / "incident_escalation.md",
        base_dir / "mock_data" / "logs" / "security_logs.csv",
        base_dir / "mock_data" / "rbac_config.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ {file_path.name} exists")
        else:
            print(f"❌ {file_path.name} missing")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files exist!")
    else:
        print("\n❌ Some required files are missing")
    
    return all_exist

def test_environment():
    """Test environment configuration"""
    print("\nTesting environment...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✓ OPENAI_API_KEY is set")
        if openai_key.startswith("sk-"):
            print("✓ OPENAI_API_KEY format looks correct")
        else:
            print("⚠️  OPENAI_API_KEY format might be incorrect")
    else:
        print("❌ OPENAI_API_KEY not set")
        print("   Please create a .env file with your OpenAI API key")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 Testing Security Assistant Backend Setup\n")
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Environment Test", test_environment)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        if not test_func():
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! Backend setup looks good.")
        print("\nNext steps:")
        print("1. Make sure you have OPENAI_API_KEY in your .env file")
        print("2. Run: python -m app.main")
        print("3. Test the API at http://localhost:8000")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print('='*50)

if __name__ == "__main__":
    main()
