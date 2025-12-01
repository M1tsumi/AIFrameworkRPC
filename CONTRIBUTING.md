# Contributing to AIFrameworkRPC

Thank you for your interest in contributing to AIFrameworkRPC! This document provides guidelines for contributors.

## 🤝 How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](https://github.com/yourusername/ai-framework-rpc/issues) page
- Provide detailed information about the bug
- Include steps to reproduce the issue
- Specify your operating system, Python version, and library versions

### Suggesting Features

- Open an issue with the "enhancement" label
- Clearly describe the feature you'd like to see
- Explain why this feature would be useful
- Consider if it fits the project's scope

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or poetry

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-framework-rpc.git
cd ai-framework-rpc
```

2. Install in development mode:
```bash
pip install -e .[dev]
```

3. Create a configuration file:
```bash
python -m ai_framework_rpc.cli config --create
```

4. Set your Discord client ID:
```bash
# Edit ai_rpc_config.json or set environment variable
export DISCORD_CLIENT_ID="your_client_id_here"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_framework_rpc

# Run specific test file
pytest tests/test_core.py
```

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run these tools before submitting:
```bash
black ai_framework_rpc/
flake8 ai_framework_rpc/
mypy ai_framework_rpc/
```

## 📁 Project Structure

```
ai_framework_rpc/
├── ai_framework_rpc/          # Main package
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Core AIFrameworkRPC class
│   ├── integrations.py       # AI tool integrations
│   ├── multi_tool.py         # Multi-tool management
│   ├── config.py             # Configuration management
│   └── cli.py                # Command line interface
├── examples/                 # Usage examples
│   ├── stable_diffusion.py   # Stable Diffusion example
│   ├── llm_chat.py          # LLM chat example
│   ├── multi_tool.py        # Multi-tool example
│   └── bot_integration.py   # Bot integration example
├── tests/                   # Test suite
├── docs/                    # Documentation
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## 🎯 Contribution Areas

### High Priority

- **Additional AI Tool Integrations**: Support for more AI tools (Midjourney, DALL-E, etc.)
- **Enhanced Bot Features**: Better Discord bot integration
- **Error Handling**: Improve error handling and logging
- **Documentation**: Expand documentation and examples

### Medium Priority

- **Web Dashboard**: Web interface for status management
- **Analytics**: Usage tracking and analytics
- **Plugin System**: Allow custom integrations
- **Mobile Support**: Mobile app for remote monitoring

### Low Priority

- **UI Components**: GUI components for desktop applications
- **Performance**: Performance optimizations
- **Internationalization**: Multi-language support

## 🧪 Testing

### Writing Tests

- Write tests for new features
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (Discord API, etc.)

### Test Structure

```python
import pytest
from ai_framework_rpc import AIFrameworkRPC

def test_rpc_connection():
    """Test Discord Rich Presence connection."""
    with AIFrameworkRPC("test_client_id") as rpc:
        assert rpc.connected
        # Add more assertions
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_core.py::test_rpc_connection
```

## 📝 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstring format
- Include type hints

### Example Documentation

- Add examples for new features
- Update existing examples
- Ensure examples are tested

## 🚀 Release Process

1. Update version number in `__init__.py`
2. Update CHANGELOG.md
3. Create a Git tag
4. Build and publish to PyPI
5. Update documentation

## 🤝 Code of Conduct

Please be respectful and professional in all interactions. We welcome contributors from all backgrounds and experience levels.

## 📞 Getting Help

- Create an issue for questions
- Join our Discord community
- Check existing documentation

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Documentation

Thank you for contributing to AIFrameworkRPC! 🙏
