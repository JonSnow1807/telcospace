"""Quick test script to verify Gemini API is working."""
import os
import sys
import json
import time

# Add backend to path
sys.path.insert(0, '/app')

def test_gemini():
    """Test Gemini API connection and response time."""
    print("=" * 50)
    print("GEMINI API TEST")
    print("=" * 50)
    
    try:
        import google.generativeai as genai
        print("✓ google.generativeai imported")
    except ImportError as e:
        print(f"✗ Failed to import google.generativeai: {e}")
        return False
    
    # Try to find credentials
    creds_paths = [
        "/app/info.json",
        "/app/backend/info.json",
    ]
    
    creds_path = None
    for path in creds_paths:
        if os.path.exists(path):
            creds_path = path
            print(f"✓ Found credentials at: {path}")
            break
    
    if not creds_path:
        print("✗ No credentials file found")
        return False
    
    # Load and check credentials
    with open(creds_path, 'r') as f:
        creds = json.load(f)
    
    project_id = creds.get("project_id", "N/A")
    print(f"✓ Project ID: {project_id}")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"✓ Using API key (length: {len(api_key)})")
        genai.configure(api_key=api_key)
    else:
        print("ℹ No GEMINI_API_KEY found, trying service account...")
        try:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/generative-language"]
            )
            genai.configure(credentials=credentials)
            print("✓ Service account configured")
        except Exception as e:
            print(f"✗ Service account error: {e}")
            return False
    
    # Test with gemini-2.0-flash
    print("\n--- Testing gemini-2.0-flash ---")
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✓ Model initialized")
        
        start_time = time.time()
        response = model.generate_content("Say 'Hello, test successful!' in exactly those words.")
        elapsed = time.time() - start_time
        
        print(f"✓ Response received in {elapsed:.2f}s")
        print(f"  Response: {response.text[:100]}...")
        
    except Exception as e:
        print(f"✗ gemini-2.0-flash error: {e}")
        
        # Fallback to 1.5-flash
        print("\n--- Fallback: Testing gemini-1.5-flash ---")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            start_time = time.time()
            response = model.generate_content("Say 'Hello, test successful!' in exactly those words.")
            elapsed = time.time() - start_time
            print(f"✓ Response received in {elapsed:.2f}s")
            print(f"  Response: {response.text[:100]}...")
        except Exception as e2:
            print(f"✗ gemini-1.5-flash error: {e2}")
            return False
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_gemini()
    sys.exit(0 if success else 1)
