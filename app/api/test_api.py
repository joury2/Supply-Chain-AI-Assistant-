# test_api.py
"""
Complete API testing script
Tests all endpoints and validates responses
"""
import requests
import time
import json
import pandas as pd
from datetime import datetime
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "test-token-123456789"  # Change this

# Test results
tests_passed = 0
tests_failed = 0
test_results = []

def log_test(test_name, passed, message=""):
    """Log test result"""
    global tests_passed, tests_failed
    
    if passed:
        tests_passed += 1
        status = "✅ PASS"
        print(f"{status} - {test_name}")
    else:
        tests_failed += 1
        status = "❌ FAIL"
        print(f"{status} - {test_name}")
        if message:
            print(f"   Error: {message}")
    
    test_results.append({
        'test': test_name,
        'passed': passed,
        'message': message
    })

def test_health_check():
    """Test 1: Health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') in ['healthy', 'degraded']:
                log_test("Health Check", True)
                return True
            else:
                log_test("Health Check", False, f"Unexpected status: {data.get('status')}")
                return False
        else:
            log_test("Health Check", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        log_test("Health Check", False, str(e))
        return False

def test_authentication():
    """Test 2: Authentication"""
    try:
        # Test without token (should fail)
        response = requests.get(f"{API_BASE_URL}/api/v1/models")
        
        if response.status_code == 401:
            # Test with token (should succeed)
            response = requests.get(
                f"{API_BASE_URL}/api/v1/models",
                headers={'Authorization': f'Bearer {API_TOKEN}'}
            )
            
            if response.status_code == 200:
                log_test("Authentication", True)
                return True
            else:
                log_test("Authentication", False, f"Auth failed: {response.status_code}")
                return False
        else:
            log_test("Authentication", False, "Endpoint not protected")
            return False
    except Exception as e:
        log_test("Authentication", False, str(e))
        return False

def test_list_models():
    """Test 3: List models"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/models",
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
        
        if response.status_code == 200:
            models = response.json()
            if isinstance(models, list):
                log_test("List Models", True)
                print(f"   Found {len(models)} model(s)")
                return True, models
            else:
                log_test("List Models", False, "Invalid response format")
                return False, []
        else:
            log_test("List Models", False, f"Status: {response.status_code}")
            return False, []
    except Exception as e:
        log_test("List Models", False, str(e))
        return False, []

def create_test_dataset():
    """Create test dataset"""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'demand': [100 + i + (i % 7) * 10 for i in range(100)]
    })
    
    # Save to CSV
    df.to_csv('test_dataset.csv', index=False)
    return 'test_dataset.csv'

def test_upload_dataset():
    """Test 4: Upload dataset"""
    try:
        # Create test dataset
        file_path = create_test_dataset()
        
        # Upload
        with open(file_path, 'rb') as f:
            files = {'file': ('test_dataset.csv', f, 'text/csv')}
            response = requests.post(
                f"{API_BASE_URL}/api/v1/upload",
                files=files,
                headers={'Authorization': f'Bearer {API_TOKEN}'}
            )
        
        if response.status_code == 200:
            data = response.json()
            if 'session_id' in data and 'rows' in data:
                log_test("Upload Dataset", True)
                print(f"   Session ID: {data['session_id']}")
                print(f"   Rows: {data['rows']}")
                return True, data['session_id']
            else:
                log_test("Upload Dataset", False, "Missing fields in response")
                return False, None
        else:
            log_test("Upload Dataset", False, f"Status: {response.status_code}, {response.text}")
            return False, None
    except Exception as e:
        log_test("Upload Dataset", False, str(e))
        return False, None

def test_analyze_dataset(session_id):
    """Test 5: Analyze dataset"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze",
            json={'session_id': session_id},
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'status' in data and 'validation' in data:
                log_test("Analyze Dataset", True)
                print(f"   Status: {data['status']}")
                print(f"   Valid: {data.get('validation', {}).get('valid', False)}")
                print(f"   Models: {len(data.get('compatible_models', []))}")
                return True, data
            else:
                log_test("Analyze Dataset", False, "Missing fields in response")
                return False, None
        else:
            log_test("Analyze Dataset", False, f"Status: {response.status_code}, {response.text}")
            return False, None
    except Exception as e:
        log_test("Analyze Dataset", False, str(e))
        return False, None

def test_create_forecast(session_id):
    """Test 6: Create forecast"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/forecast",
            json={
                'session_id': session_id,
                'horizon': 30
            },
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'job_id' in data:
                log_test("Create Forecast", True)
                print(f"   Job ID: {data['job_id']}")
                return True, data['job_id']
            else:
                log_test("Create Forecast", False, "Missing job_id in response")
                return False, None
        else:
            log_test("Create Forecast", False, f"Status: {response.status_code}, {response.text}")
            return False, None
    except Exception as e:
        log_test("Create Forecast", False, str(e))
        return False, None

def test_forecast_status(job_id, wait_for_completion=True):
    """Test 7: Get forecast status"""
    try:
        max_wait = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/forecast/status/{job_id}",
                headers={'Authorization': f'Bearer {API_TOKEN}'}
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                
                print(f"   Status: {status}, Progress: {data.get('progress', 0)}%")
                
                if status == 'completed':
                    log_test("Forecast Status", True)
                    return True, data.get('result')
                elif status == 'failed':
                    log_test("Forecast Status", False, f"Job failed: {data.get('error')}")
                    return False, None
                elif not wait_for_completion:
                    log_test("Forecast Status", True)
                    return True, None
                
                time.sleep(2)
            else:
                log_test("Forecast Status", False, f"Status: {response.status_code}")
                return False, None
        
        log_test("Forecast Status", False, "Timeout waiting for forecast")
        return False, None
        
    except Exception as e:
        log_test("Forecast Status", False, str(e))
        return False, None

def test_chat(session_id):
    """Test 8: Chat with AI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                'message': 'What models work with my data?',
                'session_id': session_id
            },
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'intent' in data:
                log_test("Chat", True)
                print(f"   Intent: {data['intent']}")
                print(f"   Response preview: {data['response'][:100]}...")
                return True
            else:
                log_test("Chat", False, "Missing fields in response")
                return False
        else:
            log_test("Chat", False, f"Status: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        log_test("Chat", False, str(e))
        return False

def test_delete_session(session_id):
    """Test 9: Delete session"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/session/{session_id}",
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
        
        if response.status_code == 200:
            log_test("Delete Session", True)
            return True
        else:
            log_test("Delete Session", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        log_test("Delete Session", False, str(e))
        return False

def test_rate_limiting():
    """Test 10: Rate limiting"""
    try:
        # Make many requests quickly
        responses = []
        for i in range(150):  # Exceed the limit
            response = requests.get(
                f"{API_BASE_URL}/api/v1/models",
                headers={'Authorization': f'Bearer {API_TOKEN}'}
            )
            responses.append(response.status_code)
            
            if response.status_code == 429:
                log_test("Rate Limiting", True)
                print(f"   Rate limit triggered after {i+1} requests")
                return True
        
        log_test("Rate Limiting", False, "Rate limit not enforced")
        return False
        
    except Exception as e:
        log_test("Rate Limiting", False, str(e))
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    global tests_passed, tests_failed
    
    print("="*70)
    print("🧪 FASTAPI TESTING SUITE")
    print("="*70)
    print(f"API URL: {API_BASE_URL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n❌ API is not healthy. Stopping tests.")
        return
    
    # Test 2: Authentication
    print("\n2️⃣ Testing Authentication...")
    auth_ok = test_authentication()
    
    if not auth_ok:
        print("\n⚠️ Authentication issues detected")
    
    # Test 3: List Models
    print("\n3️⃣ Testing List Models...")
    models_ok, models = test_list_models()
    
    # Test 4: Upload Dataset
    print("\n4️⃣ Testing Upload Dataset...")
    upload_ok, session_id = test_upload_dataset()
    
    if not upload_ok:
        print("\n❌ Upload failed. Skipping dependent tests.")
        print_summary()
        return
    
    # Test 5: Analyze Dataset
    print("\n5️⃣ Testing Analyze Dataset...")
    analyze_ok, analysis = test_analyze_dataset(session_id)
    
    # Test 6: Create Forecast
    print("\n6️⃣ Testing Create Forecast...")
    forecast_ok, job_id = test_create_forecast(session_id)
    
    if forecast_ok and job_id:
        # Test 7: Forecast Status
        print("\n7️⃣ Testing Forecast Status...")
        status_ok, result = test_forecast_status(job_id, wait_for_completion=True)
    
    # Test 8: Chat
    print("\n8️⃣ Testing Chat...")
    chat_ok = test_chat(session_id)
    
    # Test 9: Delete Session
    print("\n9️⃣ Testing Delete Session...")
    delete_ok = test_delete_session(session_id)
    
    # Test 10: Rate Limiting (optional - slow)
    # print("\n🔟 Testing Rate Limiting...")
    # rate_ok = test_rate_limiting()
    
    # Print summary
    print_summary()

def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    total = tests_passed + tests_failed
    success_rate = (tests_passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if tests_failed > 0:
        print("\n❌ Failed Tests:")
        for result in test_results:
            if not result['passed']:
                print(f"  - {result['test']}: {result['message']}")
    
    print("="*70)
    
    # Exit code
    if tests_failed == 0:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {tests_failed} test(s) failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        print_summary()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)