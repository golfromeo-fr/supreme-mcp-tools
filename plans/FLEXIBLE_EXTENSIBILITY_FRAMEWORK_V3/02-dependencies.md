# Dependencies and Requirements

## Required Python Packages

Create a `requirements_fef_v3.txt` file with the following dependencies:

```txt
# Core HTTP and async support
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
aiohttp>=3.9.0
websockets>=12.0

# Data validation and serialization
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Database support (for SQLite persistence)
aiosqlite>=0.19.0

# Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Monitoring and observability
prometheus-client>=0.19.0
structlog>=23.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
httpx>=0.25.0  # For testing FastAPI

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.7.0
```

## Installation

```bash
# Install all dependencies
pip install -r requirements_fef_v3.txt

# Or install individually
pip install fastapi uvicorn aiohttp websockets pydantic
```

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM per tool | 512MB | 1GB |
| Disk space | 100MB | 500MB |
| Network | Localhost | Low-latency LAN |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✅ Full | Primary development platform |
| macOS | ✅ Full | Tested on Intel and Apple Silicon |
| Windows | ✅ Full | Requires Windows 10+ |
| Docker | ✅ Supported | See `docker/` directory |

## Optional Dependencies

### For SQLite Persistence

```txt
aiosqlite>=0.19.0
```

### For Redis-based Distributed Deployment

```txt
redis>=5.0.0
hiredis>=2.2.0
```

### For Prometheus Monitoring

```txt
prometheus-client>=0.19.0
```

### For Grafana Integration

No additional Python packages required. Use Prometheus exporter.

## Development Dependencies

```txt
# Development
black>=23.0.0
ruff>=0.1.0
mypy>=1.5.0
pre-commit>=3.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
factory-boy>=3.3.0
```

## Version Compatibility

| Package | Minimum Version | Notes |
|---------|-----------------|-------|
| Python | 3.9.0 | Required for asyncio improvements |
| FastAPI | 0.104.0 | Required for modern Pydantic |
| Pydantic | 2.5.0 | Required for V2 features |
| aiohttp | 3.9.0 | Required for connection pooling |

## Dependency Management

### Using pip-tools

```bash
# Install pip-tools
pip install pip-tools

# Compile requirements
pip-compile requirements.in

# Install from compiled requirements
pip install -r requirements.txt

# Upgrade dependencies
pip-compile --upgrade requirements.in
```

### Using Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Add new dependency
poetry add fastapi
```

### Using conda

```bash
# Create environment
conda create -n fef python=3.11

# Activate environment
conda activate fef

# Install dependencies
conda install -c conda-forge fastapi uvicorn aiohttp
```
