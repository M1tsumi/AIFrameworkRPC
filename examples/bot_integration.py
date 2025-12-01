#!/usr/bin/env python3
"""
Bot integration example for AIFrameworkRPC with Discord bot features.
"""

import asyncio
import time
import os
from ai_framework_rpc import AIFrameworkRPC, Config

class DiscordBotRPC:
    """
    Example Discord bot with AIFrameworkRPC integration.
    """
    
    def __init__(self):
        self.config = Config()
        self.rpc = AIFrameworkRPC(
            discord_client_id=self.config.get_discord_client_id(),
            default_status="Bot is running"
        )
        self.bot_token = self.config.get_bot_token()
        
    async def run_bot_simulation(self):
        """
        Simulate bot operations with RPC updates.
        Note: This is a simulation. Actual Discord bot would use discord.py
        """
        if not self.bot_token:
            print("Bot token not configured. Set DISCORD_BOT_TOKEN in config or environment.")
            return
        
        print("Starting bot simulation...")
        
        # Connect RPC
        if not self.rpc.connect():
            print("Failed to connect RPC")
            return
        
        try:
            # Simulate bot startup
            self.rpc.update_status(
                activity="Bot starting up",
                details="Initializing systems",
                state="Loading modules..."
            )
            time.sleep(2)
            
            # Bot is ready
            self.rpc.update_status(
                activity="Bot is online",
                details="Ready to serve the community",
                state="Monitoring channels"
            )
            print("Bot is online!")
            
            # Simulate various bot activities
            activities = [
                {
                    "activity": "Moderating chat",
                    "details": "Keeping community safe",
                    "state": "Active in 5 channels"
                },
                {
                    "activity": "Processing commands",
                    "details": "Handling user requests",
                    "state=": "15 commands/min"
                },
                {
                    "activity": "Generating AI content",
                    "details": "Creating images with AI",
                    "state": "Processing queue: 3 requests"
                },
                {
                    "activity": "Chatting with users",
                    "details": "AI-powered conversations",
                    "state": "Active in DMs"
                }
            ]
            
            for i, activity in enumerate(activities, 1):
                print(f"\nActivity {i}: {activity['activity']}")
                self.rpc.update_status(**activity)
                time.sleep(5)
                
                # Simulate sharing content
                if activity['activity'] == "Generating AI content":
                    await self.simulate_content_sharing()
            
            # Bot maintenance
            self.rpc.update_status(
                activity="Performing maintenance",
                details="Optimizing performance",
                state="Temporary slowdown"
            )
            time.sleep(3)
            
            # Back to normal
            self.rpc.update_status(
                activity="Bot is online",
                details="Ready to serve the community",
                state="All systems operational"
            )
            
            print("Bot simulation complete!")
            
        except KeyboardInterrupt:
            print("\nBot simulation interrupted")
        finally:
            self.rpc.disconnect()
            print("Bot disconnected")
    
    async def simulate_content_sharing(self):
        """Simulate sharing AI-generated content to channels."""
        if not self.config.is_auto_share_enabled():
            print("Auto-share not enabled")
            return
        
        channel_id = self.config.get_share_channel_id()
        
        # Simulate different types of content
        content_types = [
            {
                "type": "image",
                "content": "Check out this AI-generated masterpiece! 🎨",
                "file_path": "ai_art.png"
            },
            {
                "type": "text",
                "content": "Here's an AI-written poem about technology:\n\nIn circuits deep and code so bright,\nA digital mind takes flight.\nThrough networks vast and data streams,\nThe future flows in digital dreams.",
                "file_path": None
            }
        ]
        
        for content in content_types:
            print(f"Sharing {content['type']} content...")
            self.rpc.share_to_channel(
                content=content['content'],
                channel_id=channel_id,
                image_path=content.get('file_path')
            )
            time.sleep(2)

class EventDrivenBot:
    """
    Example of event-driven bot with RPC integration.
    """
    
    def __init__(self):
        self.config = Config()
        self.rpc = AIFrameworkRPC(
            discord_client_id=self.config.get_discord_client_id(),
            default_status="Event-driven bot"
        )
        
        # Register event handlers
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Setup event handlers for bot activities."""
        
        @self.rpc.on_event("user_join")
        def on_user_join(username, user_id):
            print(f"User {username} joined the server")
            self.rpc.update_status(
                activity="Welcoming new user",
                details=f"{username} just joined!",
                state="Growing community 🎉"
            )
        
        @self.rpc.on_event("command_used")
        def on_command_used(command, username):
            print(f"Command '{command}' used by {username}")
            self.rpc.update_status(
                activity="Processing commands",
                details=f"Latest: /{command} by {username}",
                state="Active bot"
            )
        
        @self.rpc.on_event("ai_generation_start")
        def on_ai_generation_start(prompt, model):
            print(f"AI generation started with {model}")
            self.rpc.update_status(
                activity="AI generation started",
                details=f"Model: {model}",
                state=f"Prompt: {prompt[:30]}..."
            )
        
        @self.rpc.on_event("ai_generation_complete")
        def on_ai_generation_complete(output_path, prompt):
            print(f"AI generation complete: {output_path}")
            self.rpc.update_status(
                activity="AI generation complete",
                details="Ready for next request",
                state="✨ Creation ready"
            )
    
    async def run_event_simulation(self):
        """Run event-driven simulation."""
        if not self.rpc.connect():
            print("Failed to connect RPC")
            return
        
        try:
            print("Starting event-driven simulation...")
            
            # Simulate events
            events = [
                ("user_join", "Alice", "123456"),
                ("command_used", "help", "Bob"),
                ("ai_generation_start", "A beautiful sunset", "Stable Diffusion"),
                ("ai_generation_complete", "sunset.png", "A beautiful sunset"),
                ("command_used", "status", "Charlie"),
                ("user_join", "David", "789012")
            ]
            
            for event in events:
                event_name = event[0]
                event_args = event[1:]
                
                print(f"Emitting event: {event_name}")
                self.rpc.emit_event(event_name, *event_args)
                time.sleep(3)
            
            print("Event simulation complete!")
            
        except KeyboardInterrupt:
            print("\nEvent simulation interrupted")
        finally:
            self.rpc.disconnect()

async def main():
    """Main function to run bot examples."""
    print("AIFrameworkRPC Bot Integration Examples")
    print("======================================")
    
    choice = input("Choose example:\n1. Basic Bot Simulation\n2. Event-Driven Bot\nEnter choice (1-2): ")
    
    if choice == "1":
        bot = DiscordBotRPC()
        await bot.run_bot_simulation()
    elif choice == "2":
        bot = EventDrivenBot()
        await bot.run_event_simulation()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
