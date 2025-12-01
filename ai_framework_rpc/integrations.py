"""
AI tool integrations for AIFrameworkRPC.
"""

import time
import os
import logging
from typing import Optional, Dict, Any
from .core import AIFrameworkRPC


class StableDiffusionRPC(AIFrameworkRPC):
    """
    Discord Rich Presence integration for Stable Diffusion.
    """
    
    def __init__(self, discord_client_id: str, model_name: str = "Stable Diffusion"):
        """
        Initialize Stable Diffusion RPC.
        
        Args:
            discord_client_id: Discord application client ID
            model_name: Name of the Stable Diffusion model
        """
        super().__init__(discord_client_id, f"Using {model_name}")
        self.model_name = model_name
        self.generation_start_time = None
        self.current_step = 0
        self.total_steps = 0
        
    def start_generation(self, prompt: str, steps: int = 20, width: int = 512, height: int = 512):
        """
        Start a new image generation.
        
        Args:
            prompt: Generation prompt
            steps: Number of generation steps
            width: Image width
            height: Image height
        """
        self.generation_start_time = time.time()
        self.current_step = 0
        self.total_steps = steps
        
        # Truncate prompt for display
        display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        
        self.update_status(
            activity="Generating art",
            details=f"Model: {self.model_name}",
            state=f"Prompt: {display_prompt} | {width}x{height}",
            large_image_key="stable_diffusion",
            large_image_text=self.model_name
        )
        
        self.emit_event("generation_start", prompt, self.model_name)
        
    def update_progress(self, step: int, total_steps: int = None):
        """
        Update generation progress.
        
        Args:
            step: Current step
            total_steps: Total steps (overrides initial value)
        """
        self.current_step = step
        if total_steps:
            self.total_steps = total_steps
            
        progress_percent = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        self.update_status(
            activity="Generating art",
            details=f"Model: {self.model_name}",
            state=f"Step {step}/{self.total_steps} ({progress_percent:.1f}%)"
        )
        
        self.emit_event("progress_update", step, self.total_steps)
        
    def complete_generation(self, output_path: str, prompt: str = ""):
        """
        Mark generation as complete.
        
        Args:
            output_path: Path to generated image
            prompt: Original prompt
        """
        generation_time = time.time() - self.generation_start_time if self.generation_start_time else 0
        
        self.update_status(
            activity="Generation complete",
            details=f"Model: {self.model_name}",
            state=f"Completed in {generation_time:.1f}s",
            large_image_key="stable_diffusion",
            large_image_text=self.model_name
        )
        
        self.emit_event("generation_complete", output_path, prompt)
        
    def share_to_channel(self, image_path: str, channel_id: str, prompt: str = "", 
                         negative_prompt: str = "", steps: int = 20, cfg_scale: float = 7.0):
        """
        Share generated image to Discord channel.
        
        Args:
            image_path: Path to generated image
            channel_id: Discord channel ID
            prompt: Generation prompt
            negative_prompt: Negative prompt
            steps: Number of steps
            cfg_scale: CFG scale used
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return
            
        # Create share message
        message = f"**🎨 New Generation**\n"
        message += f"**Model:** {self.model_name}\n"
        if prompt:
            message += f"**Prompt:** {prompt}\n"
        if negative_prompt:
            message += f"**Negative Prompt:** {negative_prompt}\n"
        message += f"**Steps:** {steps} | **CFG Scale:** {cfg_scale}"
        
        super().share_to_channel(message, channel_id, image_path)
        
    def set_model(self, model_name: str):
        """
        Change the current model.
        
        Args:
            model_name: New model name
        """
        self.model_name = model_name
        self.update_status(
            activity="Model changed",
            details=f"Now using: {model_name}",
            state="Ready to generate"
        )


class LLMRPC(AIFrameworkRPC):
    """
    Discord Rich Presence integration for Local Language Models.
    """
    
    def __init__(self, discord_client_id: str, model_name: str = "Local LLM"):
        """
        Initialize LLM RPC.
        
        Args:
            discord_client_id: Discord application client ID
            model_name: Name of the LLM model
        """
        super().__init__(discord_client_id, f"Chatting with {model_name}")
        self.model_name = model_name
        self.inference_start_time = None
        self.current_context_length = 0
        
    def start_inference(self, prompt: str, max_tokens: int = 2048):
        """
        Start LLM inference.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
        """
        self.inference_start_time = time.time()
        
        # Truncate prompt for display
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        
        self.update_status(
            activity="Chatting with AI",
            details=f"Model: {self.model_name}",
            state=f"Processing: {display_prompt}",
            large_image_key="llm",
            large_image_text=self.model_name
        )
        
        self.emit_event("inference_start", prompt, self.model_name)
        
    def update_generation(self, generated_text: str, tokens_generated: int = 0):
        """
        Update during text generation.
        
        Args:
            generated_text: Currently generated text
            tokens_generated: Number of tokens generated so far
        """
        self.current_context_length = len(generated_text)
        
        # Show first few words of generation
        display_text = generated_text[:50].replace('\n', ' ').strip()
        if len(generated_text) > 50:
            display_text += "..."
            
        self.update_status(
            activity="Generating text",
            details=f"Model: {self.model_name}",
            state=f"Generated {tokens_generated} tokens: {display_text}"
        )
        
        self.emit_event("text_update", generated_text, tokens_generated)
        
    def complete_inference(self, response: str, prompt: str = ""):
        """
        Mark inference as complete.
        
        Args:
            response: Generated response
            prompt: Original prompt
        """
        inference_time = time.time() - self.inference_start_time if self.inference_start_time else 0
        response_length = len(response)
        
        self.update_status(
            activity="Response ready",
            details=f"Model: {self.model_name}",
            state=f"Generated {response_length} chars in {inference_time:.1f}s",
            large_image_key="llm",
            large_image_text=self.model_name
        )
        
        self.emit_event("inference_complete", response, prompt)
        
    def share_to_channel(self, text: str, channel_id: str, prompt: str = "", 
                        model_name: str = None):
        """
        Share LLM response to Discord channel.
        
        Args:
            text: Generated text to share
            channel_id: Discord channel ID
            prompt: Original prompt
            model_name: Override model name
        """
        model = model_name or self.model_name
        
        # Create share message
        message = f"**💬 AI Response**\n"
        message += f"**Model:** {model}\n"
        if prompt:
            message += f"**Prompt:** {prompt}\n"
        message += f"**Response:**\n```\n{text[:1500]}```"
        
        if len(text) > 1500:
            message += f"\n...({len(text) - 1500} more characters)"
            
        super().share_to_channel(message, channel_id)
        
    def set_model(self, model_name: str):
        """
        Change the current model.
        
        Args:
            model_name: New model name
        """
        self.model_name = model_name
        self.update_status(
            activity="Model changed",
            details=f"Now using: {model_name}",
            state="Ready to chat"
        )
        
    def start_training(self, dataset_name: str, epochs: int = 10):
        """
        Start model training.
        
        Args:
            dataset_name: Name of training dataset
            epochs: Number of training epochs
        """
        self.update_status(
            activity="Training model",
            details=f"Model: {self.model_name}",
            state=f"Training on {dataset_name} for {epochs} epochs",
            large_image_key="training",
            large_image_text=f"Training {self.model_name}"
        )
        
        self.emit_event("training_start", dataset_name, epochs)
        
    def update_training_progress(self, epoch: int, loss: float = 0.0):
        """
        Update training progress.
        
        Args:
            epoch: Current epoch
            loss: Current loss value
        """
        self.update_status(
            activity="Training model",
            details=f"Model: {self.model_name}",
            state=f"Epoch {epoch} | Loss: {loss:.4f}"
        )
        
        self.emit_event("training_progress", epoch, loss)
