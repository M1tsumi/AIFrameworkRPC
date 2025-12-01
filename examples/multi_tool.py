#!/usr/bin/env python3
"""
Multi-tool integration example for AIFrameworkRPC.
"""

import time
from ai_framework_rpc import MultiToolRPC, Config

def main():
    # Load configuration
    config = Config()
    
    # Initialize multi-tool RPC
    multi_rpc = MultiToolRPC(
        discord_client_id=config.get_discord_client_id()
    )
    
    try:
        print("Connected to Discord Rich Presence")
        print(f"Available tools: {multi_rpc.list_tools()}")
        
        # Example 1: Stable Diffusion generation
        print("\n=== Stable Diffusion Generation ===")
        multi_rpc.start_stable_diffusion_generation(
            prompt="A futuristic city at sunset, cyberpunk style",
            steps=15
        )
        
        # Simulate generation progress
        for step in range(1, 16):
            time.sleep(0.3)
            multi_rpc.update_stable_diffusion_progress(step, 15)
            print(f"SD Step {step}/15")
        
        multi_rpc.complete_stable_diffusion_generation(
            output_path="cyberpunk_city.png",
            prompt="A futuristic city at sunset, cyberpunk style"
        )
        print("Stable Diffusion generation complete")
        
        # Switch to LLM
        print("\n=== Switching to LLM ===")
        multi_rpc.switch_tool("llm")
        print(f"Current tool: {multi_rpc.current_tool}")
        
        # Example 2: LLM chat
        print("\n=== LLM Conversation ===")
        multi_rpc.start_llm_inference(
            "What do you think about AI-generated art?",
            max_tokens=150
        )
        
        # Simulate text generation
        response_parts = [
            "AI-generated art represents a fascinating intersection",
            "of technology and creativity. As an AI, I find it",
            "intriguing how algorithms can produce aesthetic works",
            "that evoke human emotions and interpretations."
        ]
        
        generated_text = ""
        tokens = 0
        for part in response_parts:
            time.sleep(0.5)
            generated_text += part + " "
            tokens += len(part.split())
            multi_rpc.update_llm_generation(generated_text.strip(), tokens)
            print(f"Generated: {part}")
        
        multi_rpc.complete_llm_inference(
            response="AI-generated art represents a fascinating intersection of technology and creativity...",
            prompt="What do you think about AI-generated art?"
        )
        print("LLM conversation complete")
        
        # Example 3: Check status of all tools
        print("\n=== Tool Statuses ===")
        all_status = multi_rpc.get_all_status()
        for tool_name, status in all_status.items():
            print(f"{tool_name}: {status['current_activity']} - {status['current_state']}")
        
        # Example 4: Update all tools with same status
        print("\n=== Updating All Tools ===")
        multi_rpc.update_all_status(
            activity="Multi-tool demo complete",
            details="Successfully demonstrated all integrations",
            state="Ready for next task"
        )
        
        # Keep status for a while
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect all tools
        multi_rpc.disconnect_all()
        print("Disconnected all tools from Discord Rich Presence")

if __name__ == "__main__":
    main()
