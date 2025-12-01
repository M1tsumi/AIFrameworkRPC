"""
User-friendly setup wizard for AIFrameworkRPC.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .config import Config
from .core import AIFrameworkRPC

class SetupWizard:
    """
    Interactive setup wizard for easy configuration.
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    def run_wizard(self) -> bool:
        """
        Run the complete setup wizard.
        
        Returns:
            True if setup successful, False otherwise
        """
        print("🚀 AIFrameworkRPC Setup Wizard")
        print("=" * 40)
        print("This wizard will help you configure AIFrameworkRPC for Discord Rich Presence integration.")
        print()
        
        try:
            # Step 1: Discord Application Setup
            if not self._setup_discord_app():
                return False
            
            # Step 2: Basic Configuration
            if not self._setup_basic_config():
                return False
            
            # Step 3: Test Connection
            if not self._test_connection():
                return False
            
            # Step 4: Optional Features
            self._setup_optional_features()
            
            # Step 5: Save Configuration
            self._save_configuration()
            
            # Step 6: Success and Next Steps
            self._show_success()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled by user.")
            return False
        except Exception as e:
            print(f"\n\n❌ Setup failed: {e}")
            return False
    
    def _setup_discord_app(self) -> bool:
        """Setup Discord application configuration."""
        print("📱 Step 1: Discord Application Setup")
        print("-" * 35)
        
        print("To use AIFrameworkRPC, you need a Discord application:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Click 'New Application' and give it a name")
        print("3. Go to 'Rich Presence' and enable it")
        print("4. Copy your Application ID")
        print()
        
        # Check if client ID already exists
        existing_client_id = self.config.get_discord_client_id()
        if existing_client_id:
            use_existing = input(f"Found existing Discord Client ID: {existing_client_id}. Use it? (y/n): ").lower().strip()
            if use_existing in ['y', 'yes']:
                return True
        
        while True:
            client_id = input("Enter your Discord Application ID: ").strip()
            
            if not client_id:
                print("❌ Client ID cannot be empty.")
                continue
            
            # Basic validation (should be numeric)
            if not client_id.isdigit():
                print("❌ Client ID should be numeric. Please check your Discord application.")
                continue
            
            self.config.set("discord_client_id", client_id)
            print("✅ Discord Client ID saved!")
            return True
    
    def _setup_basic_config(self) -> bool:
        """Setup basic configuration."""
        print("\n⚙️ Step 2: Basic Configuration")
        print("-" * 30)
        
        # Default status
        default_status = self.config.get("default_status", "Working with AI tools")
        custom_status = input(f"Default status message [{default_status}]: ").strip()
        if custom_status:
            self.config.set("default_status", custom_status)
        
        # Status templates
        print("\nStatus templates (you can customize these later):")
        templates = self.config.get("status_templates", {})
        for key, template in templates.items():
            print(f"  {key}: {template}")
        
        customize = input("\nCustomize status templates? (y/n): ").lower().strip()
        if customize in ['y', 'yes']:
            self._customize_templates()
        
        print("✅ Basic configuration complete!")
        return True
    
    def _customize_templates(self):
        """Allow user to customize status templates."""
        templates = self.config.get("status_templates", {})
        
        for key in ['generating', 'training', 'chatting', 'idle']:
            current = templates.get(key, "")
            new_template = input(f"{key} template [{current}]: ").strip()
            if new_template:
                templates[key] = new_template
        
        self.config.set("status_templates", templates)
    
    def _test_connection(self) -> bool:
        """Test Discord Rich Presence connection."""
        print("\n🔗 Step 3: Test Connection")
        print("-" * 25)
        
        print("Testing Discord Rich Presence connection...")
        print("Make sure Discord is running on your computer!")
        
        try:
            # Create test RPC instance
            client_id = self.config.get_discord_client_id()
            rpc = AIFrameworkRPC(client_id, "Testing AIFrameworkRPC")
            
            print("Connecting...", end="", flush=True)
            
            if rpc.connect():
                print(" ✅ Connected!")
                
                print("Updating status...", end="", flush=True)
                rpc.update_status(
                    activity="AIFrameworkRPC Test",
                    details="Setup wizard in progress",
                    state="Testing connection"
                )
                print(" ✅ Status updated!")
                
                print("Please check your Discord status - it should show 'AIFrameworkRPC Test'")
                input("Press Enter to continue...")
                
                rpc.clear_status()
                rpc.disconnect()
                
                print("✅ Connection test successful!")
                return True
            else:
                print(" ❌ Failed to connect")
                print("\nTroubleshooting tips:")
                print("1. Make sure Discord is running")
                print("2. Check that Rich Presence is enabled in your Discord application")
                print("3. Verify your Application ID is correct")
                print("4. Try restarting Discord")
                
                retry = input("\nRetry connection test? (y/n): ").lower().strip()
                return retry in ['y', 'yes']
                
        except Exception as e:
            print(f" ❌ Connection test failed: {e}")
            retry = input("\nRetry connection test? (y/n): ").lower().strip()
            return retry in ['y', 'yes']
    
    def _setup_optional_features(self):
        """Setup optional features."""
        print("\n🎯 Step 4: Optional Features")
        print("-" * 26)
        
        # Auto-share feature
        auto_share = input("Enable auto-sharing to Discord channels? (requires bot token) (y/n): ").lower().strip()
        if auto_share in ['y', 'yes']:
            self._setup_auto_share()
        
        # Performance settings
        perf = input("Configure performance settings? (y/n): ").lower().strip()
        if perf in ['y', 'yes']:
            self._setup_performance()
        
        print("✅ Optional features configured!")
    
    def _setup_auto_share(self):
        """Setup auto-share feature."""
        print("\n📤 Auto-Share Configuration")
        print("-" * 28)
        
        bot_token = input("Enter Discord Bot Token (or leave empty to set later): ").strip()
        if bot_token:
            self.config.set("bot_token", bot_token)
        
        channel_id = input("Enter Discord Channel ID for auto-sharing: ").strip()
        if channel_id:
            self.config.set("auto_share.enabled", True)
            self.config.set("auto_share.channel_id", channel_id)
            
            share_images = input("Share images automatically? (y/n): ").lower().strip()
            self.config.set("auto_share.share_images", share_images in ['y', 'yes'])
            
            share_text = input("Share text responses automatically? (y/n): ").lower().strip()
            self.config.set("auto_share.share_text", share_text in ['y', 'yes'])
        
        print("✅ Auto-share configured!")
    
    def _setup_performance(self):
        """Setup performance settings."""
        print("\n⚡ Performance Configuration")
        print("-" * 29)
        
        # Connection timeout
        current_timeout = self.config.get("timeouts.connection_timeout", 30)
        timeout = input(f"Connection timeout in seconds [{current_timeout}]: ").strip()
        if timeout and timeout.isdigit():
            self.config.set("timeouts.connection_timeout", int(timeout))
        
        # Update interval
        current_interval = self.config.get("timeouts.update_interval", 5)
        interval = input(f"Status update interval in seconds [{current_interval}]: ").strip()
        if interval and interval.isdigit():
            self.config.set("timeouts.update_interval", int(interval))
        
        print("✅ Performance settings configured!")
    
    def _save_configuration(self):
        """Save configuration to file."""
        print("\n💾 Step 5: Save Configuration")
        print("-" * 30)
        
        config_file = self.config.config_file
        
        # Create config directory if needed
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.config.save_config()
        print(f"✅ Configuration saved to: {config_file}")
        
        # Set environment variable suggestion
        print(f"\n💡 Tip: You can also set environment variables:")
        print(f"   export DISCORD_CLIENT_ID={self.config.get_discord_client_id()}")
        if self.config.get_bot_token():
            print(f"   export DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN")
    
    def _show_success(self):
        """Show success message and next steps."""
        print("\n🎉 Setup Complete!")
        print("=" * 20)
        
        print("✅ AIFrameworkRPC is now configured and ready to use!")
        print()
        
        print("🚀 Quick Start:")
        print("```python")
        print("from ai_framework_rpc import AIFrameworkRPC")
        print()
        print("# Using your configuration")
        print("rpc = AIFrameworkRPC.from_config()")
        print("rpc.connect()")
        print("rpc.update_status('Working with AI tools')")
        print("```")
        print()
        
        print("📚 Next Steps:")
        print("1. Check out the examples in the 'examples/' directory")
        print("2. Read the full documentation in README.md")
        print("3. Try the CLI tool: python -m ai_framework_rpc.cli test")
        print()
        
        print("🤝 Need Help?")
        print("- GitHub Issues: https://github.com/yourusername/ai-framework-rpc/issues")
        print("- Discord Community: https://discord.gg/yourcommunity")
        print("- Documentation: https://ai-framework-rpc.readthedocs.io/")
    
    def quick_setup(self, client_id: str, **kwargs) -> bool:
        """
        Quick setup for programmatic use.
        
        Args:
            client_id: Discord client ID
            **kwargs: Additional configuration options
            
        Returns:
            True if setup successful
        """
        self.config.set("discord_client_id", client_id)
        
        for key, value in kwargs.items():
            self.config.set(key, value)
        
        # Test connection
        try:
            rpc = AIFrameworkRPC(client_id, "Quick Setup Test")
            if rpc.connect():
                rpc.disconnect()
                self.config.save_config()
                return True
        except Exception:
            pass
        
        return False


def run_setup_wizard():
    """Run the setup wizard from command line."""
    wizard = SetupWizard()
    success = wizard.run_wizard()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    run_setup_wizard()
