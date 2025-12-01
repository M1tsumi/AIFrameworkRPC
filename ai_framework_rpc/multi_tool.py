"""
Multi-tool management for AIFrameworkRPC.
"""

import time
import logging
from typing import Dict, Any, Optional, Type, Union
from .core import AIFrameworkRPC
from .integrations import StableDiffusionRPC, LLMRPC


class MultiToolRPC:
    """
    Manager for multiple AI tool RPC integrations.
    """
    
    def __init__(self, discord_client_id: str, tools: Dict[str, Type[AIFrameworkRPC]] = None):
        """
        Initialize multi-tool RPC manager.
        
        Args:
            discord_client_id: Discord application client ID
            tools: Dictionary of tool name to tool class
        """
        self.client_id = discord_client_id
        self.tools: Dict[str, AIFrameworkRPC] = {}
        self.current_tool: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Register default tools
        if tools is None:
            tools = {
                "stable_diffusion": StableDiffusionRPC,
                "llm": LLMRPC
            }
        
        # Initialize tools
        for tool_name, tool_class in tools.items():
            self.register_tool(tool_name, tool_class)
        
        # Connect all tools
        self.connect_all()
    
    def register_tool(self, name: str, tool_class: Type[AIFrameworkRPC], **kwargs):
        """
        Register a new tool.
        
        Args:
            name: Tool name
            tool_class: Tool class
            **kwargs: Additional arguments for tool initialization
        """
        try:
            tool_instance = tool_class(self.client_id, **kwargs)
            self.tools[name] = tool_instance
            self.logger.info(f"Registered tool: {name}")
        except Exception as e:
            self.logger.error(f"Failed to register tool {name}: {e}")
    
    def connect_all(self) -> bool:
        """
        Connect all registered tools.
        
        Returns:
            True if all tools connected successfully
        """
        all_connected = True
        for name, tool in self.tools.items():
            if not tool.connect():
                self.logger.error(f"Failed to connect tool: {name}")
                all_connected = False
            else:
                self.logger.info(f"Connected tool: {name}")
        
        return all_connected
    
    def disconnect_all(self):
        """Disconnect all tools."""
        for name, tool in self.tools.items():
            tool.disconnect()
            self.logger.info(f"Disconnected tool: {name}")
    
    def switch_tool(self, tool_name: str) -> bool:
        """
        Switch to a different tool.
        
        Args:
            tool_name: Name of tool to switch to
            
        Returns:
            True if switch successful
        """
        if tool_name not in self.tools:
            self.logger.error(f"Tool not found: {tool_name}")
            return False
        
        # Clear current tool status
        if self.current_tool and self.current_tool in self.tools:
            self.tools[self.current_tool].clear_status()
        
        # Switch to new tool
        self.current_tool = tool_name
        self.logger.info(f"Switched to tool: {tool_name}")
        return True
    
    def get_current_tool(self) -> Optional[AIFrameworkRPC]:
        """
        Get the currently active tool.
        
        Returns:
            Current tool instance or None
        """
        if self.current_tool and self.current_tool in self.tools:
            return self.tools[self.current_tool]
        return None
    
    def update_status(self, activity: str, details: str = "", state: str = "", **kwargs):
        """
        Update status on current tool.
        
        Args:
            activity: Main activity text
            details: Details text
            state: State text
            **kwargs: Additional arguments
        """
        current_tool = self.get_current_tool()
        if current_tool:
            current_tool.update_status(activity, details, state, **kwargs)
        else:
            self.logger.warning("No current tool selected")
    
    def update_all_status(self, activity: str, details: str = "", state: str = "", **kwargs):
        """
        Update status on all tools.
        
        Args:
            activity: Main activity text
            details: Details text
            state: State text
            **kwargs: Additional arguments
        """
        for name, tool in self.tools.items():
            tool.update_status(f"[{name}] {activity}", details, state, **kwargs)
    
    def get_tool(self, tool_name: str) -> Optional[AIFrameworkRPC]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of tool
            
        Returns:
            Tool instance or None
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> list:
        """
        List all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tool_status(self, tool_name: str = None) -> Dict[str, Any]:
        """
        Get status information for a tool or current tool.
        
        Args:
            tool_name: Name of tool (if None, uses current tool)
            
        Returns:
            Status dictionary
        """
        if tool_name is None:
            tool_name = self.current_tool
            
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        tool = self.tools[tool_name]
        return {
            "name": tool_name,
            "connected": tool.connected,
            "current_activity": tool.current_activity,
            "current_details": tool.current_details,
            "current_state": tool.current_state,
            "start_time": tool.start_time
        }
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all tools.
        
        Returns:
            Dictionary of tool statuses
        """
        return {name: self.get_tool_status(name) for name in self.tools.keys()}
    
    # Tool-specific convenience methods
    
    def start_stable_diffusion_generation(self, prompt: str, **kwargs):
        """Start Stable Diffusion generation."""
        if self.switch_tool("stable_diffusion"):
            sd_tool = self.get_tool("stable_diffusion")
            if sd_tool and isinstance(sd_tool, StableDiffusionRPC):
                sd_tool.start_generation(prompt, **kwargs)
    
    def update_stable_diffusion_progress(self, step: int, total_steps: int = None):
        """Update Stable Diffusion progress."""
        sd_tool = self.get_tool("stable_diffusion")
        if sd_tool and isinstance(sd_tool, StableDiffusionRPC):
            sd_tool.update_progress(step, total_steps)
    
    def complete_stable_diffusion_generation(self, output_path: str, prompt: str = ""):
        """Complete Stable Diffusion generation."""
        sd_tool = self.get_tool("stable_diffusion")
        if sd_tool and isinstance(sd_tool, StableDiffusionRPC):
            sd_tool.complete_generation(output_path, prompt)
    
    def start_llm_inference(self, prompt: str, **kwargs):
        """Start LLM inference."""
        if self.switch_tool("llm"):
            llm_tool = self.get_tool("llm")
            if llm_tool and isinstance(llm_tool, LLMRPC):
                llm_tool.start_inference(prompt, **kwargs)
    
    def update_llm_generation(self, generated_text: str, tokens_generated: int = 0):
        """Update LLM text generation."""
        llm_tool = self.get_tool("llm")
        if llm_tool and isinstance(llm_tool, LLMRPC):
            llm_tool.update_generation(generated_text, tokens_generated)
    
    def complete_llm_inference(self, response: str, prompt: str = ""):
        """Complete LLM inference."""
        llm_tool = self.get_tool("llm")
        if llm_tool and isinstance(llm_tool, LLMRPC):
            llm_tool.complete_inference(response, prompt)
    
    def share_to_all_channels(self, content: str, channel_ids: Dict[str, str], **kwargs):
        """
        Share content to multiple channels using different tools.
        
        Args:
            content: Content to share
            channel_ids: Dictionary mapping tool names to channel IDs
            **kwargs: Additional arguments
        """
        for tool_name, channel_id in channel_ids.items():
            tool = self.get_tool(tool_name)
            if tool:
                tool.share_to_channel(content, channel_id, **kwargs)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect_all()
