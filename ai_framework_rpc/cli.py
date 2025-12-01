"""
Command line interface for AIFrameworkRPC.
"""

import argparse
import sys
import time
import json
from . import AIFrameworkRPC, Config, __version__

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AIFrameworkRPC - Discord Rich Presence for AI Tools"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"AIFrameworkRPC {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--create", action="store_true", help="Create default config file")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    # Test commands
    test_parser = subparsers.add_parser("test", help="Test Discord Rich Presence")
    test_parser.add_argument("--client-id", help="Discord client ID (overrides config)")
    test_parser.add_argument("--status", default="Testing AIFrameworkRPC", help="Status message")
    
    # Demo commands
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    demo_parser.add_argument("--tool", choices=["basic", "stable-diffusion", "llm"], 
                           default="basic", help="Demo tool to use")
    demo_parser.add_argument("--duration", type=int, default=30, help="Demo duration in seconds")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "config":
        handle_config_command(args)
    elif args.command == "test":
        handle_test_command(args)
    elif args.command == "demo":
        handle_demo_command(args)

def handle_config_command(args):
    """Handle configuration commands."""
    config = Config()
    
    if args.create:
        config.create_default_config_file()
        print("Default configuration created")
    
    if args.show:
        print("Current configuration:")
        print(config)
    
    if args.validate:
        if config.validate_config():
            print("Configuration is valid ✓")
        else:
            print("Configuration is invalid ✗")
            sys.exit(1)

def handle_test_command(args):
    """Handle test commands."""
    config = Config()
    
    try:
        client_id = args.client_id or config.get_discord_client_id()
    except ValueError:
        print("Error: Discord client ID not found. Set it in config or with --client-id")
        sys.exit(1)
    
    print(f"Testing Discord Rich Presence with client ID: {client_id}")
    
    with AIFrameworkRPC(client_id, args.status) as rpc:
        if not rpc.connected:
            print("Failed to connect to Discord Rich Presence")
            sys.exit(1)
        
        print("Connected successfully! Check your Discord status.")
        print(f"Status: {args.status}")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTest completed")

def handle_demo_command(args):
    """Handle demo commands."""
    config = Config()
    
    try:
        client_id = config.get_discord_client_id()
    except ValueError:
        print("Error: Discord client ID not found. Set it in config")
        sys.exit(1)
    
    print(f"Running {args.tool} demo for {args.duration} seconds...")
    
    if args.tool == "basic":
        demo_basic(client_id, args.duration)
    elif args.tool == "stable-diffusion":
        demo_stable_diffusion(client_id, args.duration)
    elif args.tool == "llm":
        demo_llm(client_id, args.duration)

def demo_basic(client_id, duration):
    """Basic demo."""
    from . import AIFrameworkRPC
    
    with AIFrameworkRPC(client_id, "Basic Demo") as rpc:
        activities = [
            ("Working with AI tools", "Demonstrating basic functionality", "Active"),
            ("Processing data", "Analyzing inputs", "Working"),
            ("Generating output", "Creating results", "Processing"),
        ]
        
        start_time = time.time()
        i = 0
        
        while time.time() - start_time < duration:
            activity, details, state = activities[i % len(activities)]
            rpc.update_status(activity, details, state)
            print(f"Status: {activity} - {details} - {state}")
            
            time.sleep(5)
            i += 1

def demo_stable_diffusion(client_id, duration):
    """Stable Diffusion demo."""
    from .integrations import StableDiffusionRPC
    
    with StableDiffusionRPC(client_id, "Stable Diffusion XL") as rpc:
        prompts = [
            "A beautiful landscape with mountains",
            "Cyberpunk city at night",
            "Fantasy dragon in ancient forest",
        ]
        
        start_time = time.time()
        prompt_index = 0
        
        while time.time() - start_time < duration:
            prompt = prompts[prompt_index % len(prompts)]
            steps = 20
            
            print(f"Generating: {prompt}")
            rpc.start_generation(prompt, steps=steps)
            
            # Simulate progress
            for step in range(1, steps + 1):
                rpc.update_progress(step, steps)
                time.sleep(0.5)
            
            rpc.complete_generation("output.png", prompt)
            print("Generation complete!")
            
            time.sleep(3)
            prompt_index += 1

def demo_llm(client_id, duration):
    """LLM demo."""
    from .integrations import LLMRPC
    
    with LLMRPC(client_id, "Llama 2 7B") as rpc:
        prompts = [
            "Tell me about artificial intelligence",
            "What are the benefits of machine learning?",
            "Explain neural networks",
        ]
        
        start_time = time.time()
        prompt_index = 0
        
        while time.time() - start_time < duration:
            prompt = prompts[prompt_index % len(prompts)]
            
            print(f"Processing: {prompt}")
            rpc.start_inference(prompt)
            
            # Simulate text generation
            response = "This is a simulated response from the AI model..."
            rpc.update_generation(response, len(response.split()))
            time.sleep(2)
            
            rpc.complete_inference(response, prompt)
            print("Response complete!")
            
            time.sleep(3)
            prompt_index += 1

if __name__ == "__main__":
    main()
