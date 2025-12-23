"""Quick test for LLM floor plan processing."""
import sys
import time
import logging

sys.path.insert(0, '/app')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_processing():
    """Test the LLM floor plan processing pipeline."""
    
    # Find a test image
    import os
    upload_dir = "/app/static/uploads"
    test_images = [f for f in os.listdir(upload_dir) if f.endswith(('.webp', '.png', '.jpg'))]
    
    if not test_images:
        print("No test images found!")
        return False
    
    test_image = os.path.join(upload_dir, test_images[0])
    print(f"\n=== Testing LLM Floor Plan Processing ===")
    print(f"Test image: {test_image}")
    
    # Test the LLM agent directly
    print("\n--- Testing LLM Agent Directly ---")
    try:
        from app.services.llm_floor_plan_agent import GeminiFloorPlanAgent
        
        start_time = time.time()
        agent = GeminiFloorPlanAgent()
        init_time = time.time() - start_time
        print(f"✓ Agent initialized in {init_time:.2f}s")
        
        start_time = time.time()
        result = agent.process(test_image)
        process_time = time.time() - start_time
        
        if result and result.get("map_data"):
            map_data = result["map_data"]
            walls = map_data.get("walls", [])
            print(f"✓ Processed in {process_time:.2f}s")
            print(f"  Walls detected: {len(walls)}")
            if walls:
                print(f"  First wall: {walls[0]}")
        else:
            print(f"✗ No map_data returned")
            if result and result.get("error"):
                print(f"  Error: {result['error']}")
                
    except Exception as e:
        import traceback
        print(f"✗ LLM Agent test failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n=== Test Complete ===")
    return True


if __name__ == "__main__":
    success = test_llm_processing()
    sys.exit(0 if success else 1)
