#!/usr/bin/env python3
"""
LLM chat integration example for AIFrameworkRPC.
"""

import time
from ai_framework_rpc import LLMRPC, Config

def main():
    # Load configuration
    config = Config()
    
    # Initialize LLM RPC
    llm_rpc = LLMRPC(
        discord_client_id=config.get_discord_client_id(),
        model_name="Llama 2 7B Chat"
    )
    
    try:
        # Connect to Discord
        if not llm_rpc.connect():
            print("Failed to connect to Discord Rich Presence")
            return
        
        print("Connected to Discord Rich Presence")
        
        # Example conversations
        conversations = [
            "Tell me about artificial intelligence",
            "What are the benefits of machine learning?",
            "Explain neural networks in simple terms",
            "How does natural language processing work?"
        ]
        
        for i, prompt in enumerate(conversations, 1):
            print(f"\n--- Conversation {i} ---")
            print(f"User: {prompt}")
            
            # Start inference
            llm_rpc.start_inference(prompt, max_tokens=2048)
            
            # Simulate text generation
            responses = [
                "Artificial intelligence is a fascinating field of computer science...",
                "Machine learning offers numerous benefits including automation...",
                "Neural networks are computing systems inspired by biological neural networks...",
                "Natural language processing enables computers to understand and generate human language..."
            ]
            
            response = responses[i-1]
            
            # Simulate streaming generation
            words = response.split()
            generated_text = ""
            tokens_generated = 0
            
            for j, word in enumerate(words):
                time.sleep(0.1)  # Simulate generation time
                generated_text += word + " "
                tokens_generated += 1
                
                # Update every few words
                if j % 3 == 0:
                    llm_rpc.update_generation(generated_text.strip(), tokens_generated)
                    print(f"Generated: {generated_text.strip()[:50]}...")
            
            # Complete inference
            llm_rpc.complete_inference(response, prompt)
            print(f"AI: {response}")
            
            # Share to channel if configured
            if config.is_auto_share_enabled():
                channel_id = config.get_share_channel_id()
                llm_rpc.share_to_channel(
                    text=response,
                    channel_id=channel_id,
                    prompt=prompt
                )
                print(f"Shared to channel: {channel_id}")
            
            # Pause between conversations
            time.sleep(3)
        
        print("\nAll conversations complete")
        
        # Keep status for a while
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        llm_rpc.disconnect()
        print("Disconnected from Discord Rich Presence")

if __name__ == "__main__":
    main()
