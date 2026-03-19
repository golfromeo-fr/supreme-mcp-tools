"""
High Availability Configuration for FEF V3

Provides configuration for HA deployments with load balancing and shared state.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionStoreType(Enum):
    """Types of session stores."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class ServiceRegistryType(Enum):
    """Types of service registries."""
    BUILTIN = "builtin"
    CONSUL = "consul"
    ETCD = "etcd"
    ZOOKEEPER = "zookeeper"


@dataclass
class RedisConfig:
    """Redis configuration for shared state."""
    url: str = "redis://localhost:6379"
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True


@dataclass
class ConsulConfig:
    """Consul configuration for service discovery."""
    url: str = "http://localhost:8500"
    token: Optional[str] = None
    datacenter: Optional[str] = None
    health_check_interval: int = 10


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    enabled: bool = True
    health_check_path: str = "/health"
    health_check_interval: int = 10
    health_check_timeout: int = 5
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2


@dataclass
class HAConfig:
    """
    High Availability configuration.
    
    Supports multiple management server instances with shared state
    and automatic failover.
    """
    # Replication
    replicas: int = 1
    instance_id: Optional[str] = None
    
    # Session store
    session_store: SessionStoreType = SessionStoreType.MEMORY
    redis_config: RedisConfig = field(default_factory=RedisConfig)
    
    # Service registry
    service_registry: ServiceRegistryType = ServiceRegistryType.BUILTIN
    consul_config: ConsulConfig = field(default_factory=ConsulConfig)
    
    # Health checks
    health_check_interval: int = 10
    failover_timeout: int = 30
    
    # Load balancer
    load_balancer: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    
    # State sync
    state_sync_interval: int = 5
    state_sync_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "replicas": self.replicas,
            "instance_id": self.instance_id,
            "session_store": self.session_store.value,
            "redis_config": asdict(self.redis_config),
            "service_registry": self.service_registry.value,
            "consul_config": asdict(self.consul_config),
            "health_check_interval": self.health_check_interval,
            "failover_timeout": self.failover_timeout,
            "load_balancer": asdict(self.load_balancer),
            "state_sync_interval": self.state_sync_interval,
            "state_sync_enabled": self.state_sync_enabled
        }
    
    def save(self, config_file: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to config file
        """
        path = Path(config_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved HA config to {path}")
    
    @classmethod
    def load(cls, config_file: str) -> "HAConfig":
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            HAConfig instance
        """
        path = Path(config_file).expanduser()
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return cls()
        
        with open(path, "r") as f:
            data = json.load(f)
        
        config = cls()
        config.replicas = data.get("replicas", 1)
        config.instance_id = data.get("instance_id")
        
        if "session_store" in data:
            config.session_store = SessionStoreType(data["session_store"])
        
        if "redis_config" in data:
            config.redis_config = RedisConfig(**data["redis_config"])
        
        if "service_registry" in data:
            config.service_registry = ServiceRegistryType(data["service_registry"])
        
        if "consul_config" in data:
            config.consul_config = ConsulConfig(**data["consul_config"])
        
        config.health_check_interval = data.get("health_check_interval", 10)
        config.failover_timeout = data.get("failover_timeout", 30)
        
        if "load_balancer" in data:
            config.load_balancer = LoadBalancerConfig(**data["load_balancer"])
        
        config.state_sync_interval = data.get("state_sync_interval", 5)
        config.state_sync_enabled = data.get("state_sync_enabled", True)
        
        logger.info(f"Loaded HA config from {path}")
        return config


class SharedStateManager:
    """
    Manages shared state across multiple management server instances.
    
    Uses Redis for distributed state synchronization.
    """
    
    def __init__(self, config: HAConfig):
        """
        Initialize the shared state manager.
        
        Args:
            config: HA configuration
        """
        self.config = config
        self._redis = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the shared state store."""
        if self.config.session_store == SessionStoreType.REDIS:
            try:
                import aioredis
                self._redis = await aioredis.from_url(
                    self.config.redis_config.url,
                    password=self.config.redis_config.password,
                    db=self.config.redis_config.db,
                    max_connections=self.config.redis_config.max_connections
                )
                self._connected = True
                logger.info("Connected to Redis for shared state")
            except ImportError:
                logger.warning("aioredis not installed, falling back to memory")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the shared state store."""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a shared state value.
        
        Args:
            key: State key
            value: State value (will be JSON serialized)
            ttl: Optional TTL in seconds
        """
        import json
        
        if self._connected and self._redis:
            data = json.dumps(value)
            if ttl:
                await self._redis.setex(key, ttl, data)
            else:
                await self._redis.set(key, data)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a shared state value.
        
        Args:
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        import json
        
        if self._connected and self._redis:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
        return default
    
    async def delete(self, key: str) -> bool:
        """
        Delete a shared state value.
        
        Args:
            key: State key
            
        Returns:
            True if deleted
        """
        if self._connected and self._redis:
            result = await self._redis.delete(key)
            return result > 0
        return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        if self._connected and self._redis:
            keys = await self._redis.keys(pattern)
            return [k.decode() for k in keys]
        return []
    
    async def publish(self, channel: str, message: Any) -> None:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message (will be JSON serialized)
        """
        import json
        
        if self._connected and self._redis:
            await self._redis.publish(channel, json.dumps(message))
    
    async def subscribe(self, channel: str):
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            Async iterator for messages
        """
        if self._connected and self._redis:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)
            return pubsub
        return None


# Default HA configuration
DEFAULT_HA_CONFIG = HAConfig(
    replicas=1,
    session_store=SessionStoreType.MEMORY,
    service_registry=ServiceRegistryType.BUILTIN,
    health_check_interval=10,
    failover_timeout=30
)
