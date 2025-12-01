"""
Tests for core AIFrameworkRPC functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from ai_framework_rpc.core import AIFrameworkRPC


class TestAIFrameworkRPC:
    """Test cases for AIFrameworkRPC core class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client_id = "123456789012345678"
        self.rpc = AIFrameworkRPC(self.client_id, "Test Status")
    
    def test_init(self):
        """Test RPC initialization."""
        assert self.rpc.client_id == self.client_id
        assert self.rpc.default_status == "Test Status"
        assert self.rpc.connected == False
        assert self.rpc.current_activity == "Test Status"
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_connect_success(self, mock_presence):
        """Test successful Discord connection."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        result = self.rpc.connect()
        
        assert result == True
        assert self.rpc.connected == True
        mock_rpc_instance.connect.assert_called_once()
        mock_rpc_instance.update.assert_called_once()
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_connect_failure(self, mock_presence):
        """Test failed Discord connection."""
        mock_rpc_instance = Mock()
        mock_rpc_instance.connect.side_effect = Exception("Connection failed")
        mock_presence.return_value = mock_rpc_instance
        
        result = self.rpc.connect()
        
        assert result == False
        assert self.rpc.connected == False
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', False)
    def test_connect_no_pypresence(self):
        """Test connection when pypresence is not available."""
        with pytest.raises(ImportError):
            self.rpc.connect()
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_update_status(self, mock_presence):
        """Test status update."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        self.rpc.connect()
        self.rpc.update_status(
            activity="New Activity",
            details="New Details",
            state="New State"
        )
        
        mock_rpc_instance.update.assert_called()
        assert self.rpc.current_activity == "New Activity"
        assert self.rpc.current_details == "New Details"
        assert self.rpc.current_state == "New State"
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_update_status_not_connected(self, mock_presence):
        """Test status update when not connected."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        self.rpc.update_status("Test Activity")
        
        # Should not call update when not connected
        mock_rpc_instance.update.assert_not_called()
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_clear_status(self, mock_presence):
        """Test clearing status."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        self.rpc.connect()
        self.rpc.clear_status()
        
        mock_rpc_instance.clear.assert_called_once()
    
    def test_event_handlers(self):
        """Test event handler registration and emission."""
        # Test decorator
        @self.rpc.on_event("test_event")
        def test_handler(data):
            return f"handled: {data}"
        
        assert "test_event" in self.rpc.event_handlers
        assert len(self.rpc.event_handlers["test_event"]) == 1
        
        # Test event emission
        with patch.object(test_handler, return_value="handled: test") as mock_handler:
            self.rpc.emit_event("test_event", "test")
            mock_handler.assert_called_once_with("test")
    
    def test_emit_event_no_handlers(self):
        """Test emitting event with no handlers."""
        # Should not raise an error
        self.rpc.emit_event("nonexistent_event", "data")
    
    @patch('ai_framework_rpc.core.DISCORD_AVAILABLE', False)
    def test_share_to_channel_no_discord(self):
        """Test sharing to channel when discord.py is not available."""
        result = self.rpc.share_to_channel("test content", "123456")
        # Should not raise an error, just log a warning
        assert result is None
    
    @patch('ai_framework_rpc.core.DISCORD_AVAILABLE', True)
    def test_share_to_channel_no_token(self):
        """Test sharing to channel without bot token."""
        with patch.dict('os.environ', {}, clear=True):
            result = self.rpc.share_to_channel("test content", "123456")
            assert result is None
    
    @patch('ai_framework_rpc.core.DISCORD_AVAILABLE', True)
    @patch.dict('os.environ', {'DISCORD_BOT_TOKEN': 'test_token'})
    def test_share_to_channel_no_channel_id(self):
        """Test sharing to channel without channel ID."""
        result = self.rpc.share_to_channel("test content")
        assert result is None
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_context_manager(self, mock_presence):
        """Test context manager functionality."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        with self.rpc as rpc:
            assert rpc.connected == True
            mock_rpc_instance.connect.assert_called_once()
        
        mock_rpc_instance.close.assert_called_once()
    
    def test_cleanup_on_deletion(self):
        """Test cleanup on object deletion."""
        with patch.object(self.rpc, 'disconnect') as mock_disconnect:
            del self.rpc
            mock_disconnect.assert_called_once()


class TestAIFrameworkRPCIntegration:
    """Integration tests for AIFrameworkRPC (with mocked external dependencies)."""
    
    @patch('ai_framework_rpc.core.PYPRESENCE_AVAILABLE', True)
    @patch('ai_framework_rpc.core.Presence')
    def test_full_workflow(self, mock_presence):
        """Test complete workflow with mocked dependencies."""
        mock_rpc_instance = Mock()
        mock_presence.return_value = mock_rpc_instance
        
        rpc = AIFrameworkRPC("test_client_id", "Test Status")
        
        # Connect
        assert rpc.connect() == True
        assert rpc.connected == True
        
        # Update status multiple times
        rpc.update_status("Activity 1", "Details 1", "State 1")
        rpc.update_status("Activity 2", "Details 2", "State 2")
        
        # Clear status
        rpc.clear_status()
        
        # Disconnect
        rpc.disconnect()
        assert rpc.connected == False
        
        # Verify all calls were made
        assert mock_rpc_instance.connect.called
        assert mock_rpc_instance.update.call_count == 3  # initial + 2 updates
        assert mock_rpc_instance.clear.called
        assert mock_rpc_instance.close.called


if __name__ == "__main__":
    pytest.main([__file__])
