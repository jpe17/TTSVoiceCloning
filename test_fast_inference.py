from infer_voice import speak_text, preload_voice
import time

def test_fast_inference():
    """Test the optimized fast inference"""
    
    print("=== Testing Optimized Fast Inference ===")
    
    # Preload the voice (this happens once)
    print("\n1. Preloading voice...")
    preload_voice("elonmusk")
    
    # Test multiple generations
    texts = [
        "Hello, this is a quick test.",
        "The inference should be much faster now.",
        "This is the third test sentence.",
        "And this is the final test."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Generating: '{text}'")
        start_time = time.time()
        
        output_file = speak_text(text, "elonmusk", play_audio=False)
        
        end_time = time.time()
        print(f"   Generation time: {end_time - start_time:.2f}s")
    
    print("\n=== Test Complete ===")
    print("Subsequent generations should be much faster!")

if __name__ == "__main__":
    test_fast_inference() 