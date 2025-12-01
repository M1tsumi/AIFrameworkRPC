#!/usr/bin/env python3
"""
Stable Diffusion integration example for AIFrameworkRPC.
"""

import time
import os
from ai_framework_rpc import StableDiffusionRPC, Config

def main():
    # Load configuration
    config = Config()
    
    # Initialize Stable Diffusion RPC
    sd_rpc = StableDiffusionRPC(
        discord_client_id=config.get_discord_client_id(),
        model_name="Stable Diffusion XL"
    )
    
    try:
        # Connect to Discord
        if not sd_rpc.connect():
            print("Failed to connect to Discord Rich Presence")
            return
        
        print("Connected to Discord Rich Presence")
        
        # Example generation
        prompt = "A beautiful landscape with mountains and a lake, digital art style"
        steps = 20
        width, height = 512, 512
        
        # Start generation
        print(f"Starting generation: {prompt}")
        sd_rpc.start_generation(prompt, steps=steps, width=width, height=height)
        
        # Simulate generation progress
        for step in range(1, steps + 1):
            time.sleep(0.5)  # Simulate processing time
            sd_rpc.update_progress(step, steps)
            print(f"Step {step}/{steps} complete")
        
        # Complete generation
        output_path = "generated_image.png"
        sd_rpc.complete_generation(output_path, prompt)
        print(f"Generation complete: {output_path}")
        
        # Share to channel if configured
        if config.is_auto_share_enabled():
            channel_id = config.get_share_channel_id()
            if os.path.exists(output_path):
                sd_rpc.share_to_channel(
                    image_path=output_path,
                    channel_id=channel_id,
                    prompt=prompt,
                    steps=steps,
                    cfg_scale=7.0
                )
                print(f"Shared to channel: {channel_id}")
            else:
                print(f"Output file not found: {output_path}")
        
        # Keep status for a while
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        sd_rpc.disconnect()
        print("Disconnected from Discord Rich Presence")

if __name__ == "__main__":
    main()
