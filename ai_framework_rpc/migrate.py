"""
Migration tools for breaking changes in AIFrameworkRPC v0.2.0
"""

import json
import shutil
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import pickle
import traceback


@dataclass
class MigrationStep:
    """A single migration step."""
    name: str
    description: str
    version_from: str
    version_to: str
    migration_func: Callable[[], bool]
    rollback_func: Optional[Callable[[], bool]] = None
    critical: bool = False  # If True, migration must succeed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version_from': self.version_from,
            'version_to': self.version_to,
            'critical': self.critical
        }


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    steps_completed: List[str]
    steps_failed: List[str]
    errors: List[str]
    duration: float
    backup_created: bool
    backup_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MigrationManager:
    """
    Manages database and configuration migrations.
    
    Features:
    - Automated migration detection
    - Backup and rollback capabilities
    - Step-by-step migration process
    - Migration validation
    - Progress tracking
    """
    
    def __init__(self, config_dir: str = ".", backup_dir: str = "backups"):
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Migration tracking
        self.migration_log_file = self.config_dir / "migration_log.json"
        self.current_version_file = self.config_dir / "current_version.txt"
        
        # Migration steps
        self.migration_steps: List[MigrationStep] = []
        self._setup_migration_steps()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load current version
        self.current_version = self._get_current_version()
    
    def _get_current_version(self) -> str:
        """Get the current version of the installation."""
        try:
            if self.current_version_file.exists():
                with open(self.current_version_file, 'r') as f:
                    return f.read().strip()
            else:
                # Try to detect from setup.py or other files
                setup_file = self.config_dir / "setup.py"
                if setup_file.exists():
                    with open(setup_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if 'version=' in line and '"' in line:
                                version = line.split('"')[1]
                                return version
                return "0.1.0"  # Default to earliest version
        except Exception as e:
            self.logger.warning(f"Could not determine current version: {e}")
            return "0.1.0"
    
    def _setup_migration_steps(self):
        """Setup all available migration steps."""
        
        def migrate_0_1_0_to_0_2_0_config_format() -> bool:
            """Migrate configuration format from 0.1.0 to 0.2.0."""
            try:
                old_config_file = self.config_dir / "ai_rpc_config.json"
                new_config_file = self.config_dir / "config_v2.json"
                
                if old_config_file.exists() and not new_config_file.exists():
                    with open(old_config_file, 'r') as f:
                        old_config = json.load(f)
                    
                    # Transform old config to new format
                    new_config = {
                        'version': '2.0',
                        'discord': {
                            'client_id': old_config.get('discord_client_id', ''),
                            'default_status': old_config.get('default_status', 'Working with AI tools')
                        },
                        'auto_share': old_config.get('auto_share', {
                            'enabled': False,
                            'channel_id': '',
                            'share_images': True,
                            'share_text': False
                        }),
                        'status_templates': old_config.get('status_templates', {
                            'generating': 'Generating {tool} with {model}',
                            'training': 'Training on {dataset}',
                            'chatting': 'Chatting with {model}',
                            'idle': 'Ready for next task'
                        }),
                        'logging': old_config.get('logging', {
                            'level': 'INFO',
                            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        }),
                        'performance': old_config.get('performance', {
                            'cache_timeout': 1.0,
                            'max_workers': 2,
                            'connection_pool_size': 5
                        }),
                        'features': {
                            'enhanced_connections': True,
                            'predictive_caching': True,
                            'plugin_system': True,
                            'error_recovery': True,
                            'performance_profiling': True,
                            'web_dashboard': True,
                            'enhanced_security': True
                        },
                        'migration': {
                            'from_version': '0.1.0',
                            'migrated_at': time.time()
                        }
                    }
                    
                    with open(new_config_file, 'w') as f:
                        json.dump(new_config, f, indent=2)
                    
                    self.logger.info("Configuration format migrated successfully")
                    return True
                
                return True  # No migration needed
                
            except Exception as e:
                self.logger.error(f"Configuration migration failed: {e}")
                return False
        
        def rollback_0_1_0_to_0_2_0_config_format() -> bool:
            """Rollback configuration format migration."""
            try:
                old_config_file = self.config_dir / "ai_rpc_config.json"
                new_config_file = self.config_dir / "config_v2.json"
                backup_file = self.backup_dir / "ai_rpc_config.json.backup"
                
                if backup_file.exists():
                    shutil.copy2(backup_file, old_config_file)
                    if new_config_file.exists():
                        new_config_file.unlink()
                    
                    self.logger.info("Configuration format rollback completed")
                    return True
                
                return False
                
            except Exception as e:
                self.logger.error(f"Configuration rollback failed: {e}")
                return False
        
        def migrate_0_1_0_to_0_2_0_cache_format() -> bool:
            """Migrate cache format from 0.1.0 to 0.2.0."""
            try:
                old_cache_dir = self.config_dir / "cache"
                new_cache_dir = self.config_dir / "cache_v2"
                
                if old_cache_dir.exists() and not new_cache_dir.exists():
                    new_cache_dir.mkdir(exist_ok=True)
                    
                    # Migrate cache files to new format
                    for cache_file in old_cache_dir.glob("*.cache"):
                        try:
                            with open(cache_file, 'rb') as f:
                                old_data = pickle.load(f)
                            
                            # Convert to new cache format
                            new_data = {
                                'key': cache_file.stem,
                                'value': old_data,
                                'created_at': time.time(),
                                'version': '2.0'
                            }
                            
                            new_cache_file = new_cache_dir / f"{cache_file.stem}.json"
                            with open(new_cache_file, 'w') as f:
                                json.dump(new_data, f, indent=2)
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to migrate cache file {cache_file}: {e}")
                    
                    self.logger.info("Cache format migrated successfully")
                    return True
                
                return True
                
            except Exception as e:
                self.logger.error(f"Cache migration failed: {e}")
                return False
        
        def migrate_0_1_0_to_0_2_0_plugin_structure() -> bool:
            """Create plugin directory structure."""
            try:
                plugin_dir = self.config_dir / "plugins"
                plugin_data_dir = plugin_dir / "data"
                
                plugin_dir.mkdir(exist_ok=True)
                plugin_data_dir.mkdir(exist_ok=True)
                
                # Create example plugin
                example_plugin = plugin_dir / "example_plugin.py"
                if not example_plugin.exists():
                    example_content = '''
"""
Example plugin for AIFrameworkRPC v0.2.0
"""

from ai_framework_rpc.plugin_system import StatusEnhancerPlugin, PluginMetadata, PluginType

class ExamplePlugin(StatusEnhancerPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Example Plugin",
            version="1.0.0",
            description="An example plugin for demonstration",
            author="AIFrameworkRPC Team",
            plugin_type=PluginType.STATUS_ENHANCER
        )
    
    def initialize(self) -> bool:
        self.context.logger.info("Example plugin initialized")
        return True
    
    def cleanup(self):
        self.context.logger.info("Example plugin cleaned up")
    
    def enhance_status(self, activity: str, details: str = "", state: str = "") -> dict:
        return {
            'activity': f"🚀 {activity}",
            'details': f"{details} • Enhanced",
            'state': state
        }
    
    def get_status_suggestions(self, context: dict) -> list:
        return [
            "Working on amazing things 🎯",
            "Creating AI magic ✨",
            "Optimizing performance ⚡"
        ]
'''
                    with open(example_plugin, 'w') as f:
                        f.write(example_content)
                
                self.logger.info("Plugin structure created successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Plugin structure migration failed: {e}")
                return False
        
        def migrate_0_1_0_to_0_2_0_security_setup() -> bool:
            """Setup security components."""
            try:
                security_dir = self.config_dir / "security"
                security_dir.mkdir(exist_ok=True)
                
                # Create security config
                security_config_file = security_dir / "security_config.json"
                if not security_config_file.exists():
                    security_config = {
                        'encryption': {
                            'algorithm': 'AES-256',
                            'key_rotation_enabled': True,
                            'key_rotation_interval': 2592000,  # 30 days
                            'use_hardware_security': False
                        },
                        'storage': {
                            'backup_enabled': True,
                            'backup_interval': 3600,
                            'max_backup_files': 10
                        },
                        'audit': {
                            'logging_enabled': True,
                            'max_log_entries': 10000
                        },
                        'session': {
                            'timeout': 3600,
                            'max_concurrent_sessions': 10
                        }
                    }
                    
                    with open(security_config_file, 'w') as f:
                        json.dump(security_config, f, indent=2)
                
                self.logger.info("Security setup completed successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Security setup migration failed: {e}")
                return False
        
        # Add migration steps
        self.migration_steps.extend([
            MigrationStep(
                name="config_format_migration",
                description="Migrate configuration format to v2.0",
                version_from="0.1.0",
                version_to="0.2.0",
                migration_func=migrate_0_1_0_to_0_2_0_config_format,
                rollback_func=rollback_0_1_0_to_0_2_0_config_format,
                critical=True
            ),
            MigrationStep(
                name="cache_format_migration",
                description="Migrate cache format to v2.0",
                version_from="0.1.0",
                version_to="0.2.0",
                migration_func=migrate_0_1_0_to_0_2_0_cache_format,
                critical=False
            ),
            MigrationStep(
                name="plugin_structure_migration",
                description="Create plugin directory structure",
                version_from="0.1.0",
                version_to="0.2.0",
                migration_func=migrate_0_1_0_to_0_2_0_plugin_structure,
                critical=False
            ),
            MigrationStep(
                name="security_setup_migration",
                description="Setup security components",
                version_from="0.1.0",
                version_to="0.2.0",
                migration_func=migrate_0_1_0_to_0_2_0_security_setup,
                critical=False
            )
        ])
    
    def needs_migration(self, target_version: str = "0.2.0") -> bool:
        """
        Check if migration is needed.
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            True if migration is needed
        """
        return self.current_version != target_version
    
    def get_pending_migrations(self, target_version: str = "0.2.0") -> List[MigrationStep]:
        """
        Get list of pending migrations.
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            List of pending migration steps
        """
        pending = []
        
        for step in self.migration_steps:
            if (step.version_from == self.current_version and 
                step.version_to == target_version):
                pending.append(step)
        
        return pending
    
    def create_backup(self) -> str:
        """
        Create a backup of current configuration and data.
        
        Returns:
            Path to backup directory
        """
        timestamp = int(time.time())
        backup_path = self.backup_dir / f"migration_backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Backup important files
        files_to_backup = [
            "ai_rpc_config.json",
            "config.json",
            "setup.py",
            "CHANGELOG.md",
            "README.md"
        ]
        
        # Backup directories
        dirs_to_backup = [
            "cache",
            "plugins",
            "security"
        ]
        
        try:
            # Backup files
            for file_name in files_to_backup:
                source_file = self.config_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, backup_path / file_name)
            
            # Backup directories
            for dir_name in dirs_to_backup:
                source_dir = self.config_dir / dir_name
                if source_dir.exists():
                    shutil.copytree(source_dir, backup_path / dir_name, 
                                  dirs_exist_ok=True)
            
            # Create backup manifest
            manifest = {
                'created_at': time.time(),
                'version': self.current_version,
                'files_backed_up': files_to_backup,
                'dirs_backed_up': dirs_to_backup
            }
            
            with open(backup_path / "backup_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Backup created at: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def migrate(self, target_version: str = "0.2.0", 
                create_backup: bool = True) -> MigrationResult:
        """
        Perform migration to target version.
        
        Args:
            target_version: Target version to migrate to
            create_backup: Whether to create backup before migration
            
        Returns:
            MigrationResult with details
        """
        start_time = time.time()
        result = MigrationResult(
            success=False,
            steps_completed=[],
            steps_failed=[],
            errors=[],
            duration=0,
            backup_created=False
        )
        
        try:
            # Check if migration is needed
            if not self.needs_migration(target_version):
                result.success = True
                result.duration = time.time() - start_time
                self.logger.info("No migration needed")
                return result
            
            # Create backup
            backup_path = None
            if create_backup:
                try:
                    backup_path = self.create_backup()
                    result.backup_created = True
                    result.backup_path = backup_path
                except Exception as e:
                    error_msg = f"Failed to create backup: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                    # Continue without backup if it's not critical
            
            # Get pending migrations
            pending_migrations = self.get_pending_migrations(target_version)
            
            if not pending_migrations:
                self.logger.warning(f"No migration steps found for {self.current_version} -> {target_version}")
                result.success = True
                result.duration = time.time() - start_time
                return result
            
            # Execute migration steps
            for step in pending_migrations:
                try:
                    self.logger.info(f"Executing migration step: {step.name}")
                    
                    if step.migration_func():
                        result.steps_completed.append(step.name)
                        self.logger.info(f"Migration step completed: {step.name}")
                    else:
                        error_msg = f"Migration step failed: {step.name}"
                        result.steps_failed.append(step.name)
                        result.errors.append(error_msg)
                        self.logger.error(error_msg)
                        
                        # If this is a critical step, rollback and abort
                        if step.critical:
                            self.logger.error("Critical migration step failed, aborting migration")
                            if backup_path:
                                self._restore_from_backup(backup_path)
                            result.duration = time.time() - start_time
                            return result
                
                except Exception as e:
                    error_msg = f"Migration step error in {step.name}: {e}"
                    result.steps_failed.append(step.name)
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    # If this is a critical step, rollback and abort
                    if step.critical:
                        self.logger.error("Critical migration step failed, aborting migration")
                        if backup_path:
                            self._restore_from_backup(backup_path)
                        result.duration = time.time() - start_time
                        return result
            
            # Update version file
            try:
                with open(self.current_version_file, 'w') as f:
                    f.write(target_version)
                self.current_version = target_version
                
                # Log migration
                self._log_migration(self.current_version, target_version, result)
                
                result.success = True
                self.logger.info(f"Migration to {target_version} completed successfully")
                
            except Exception as e:
                error_msg = f"Failed to update version file: {e}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
        
        result.duration = time.time() - start_time
        return result
    
    def rollback(self, backup_path: str) -> MigrationResult:
        """
        Rollback to a backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            MigrationResult with rollback details
        """
        start_time = time.time()
        result = MigrationResult(
            success=False,
            steps_completed=[],
            steps_failed=[],
            errors=[],
            duration=0,
            backup_created=False
        )
        
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                result.errors.append(f"Backup path does not exist: {backup_path}")
                result.duration = time.time() - start_time
                return result
            
            # Load backup manifest
            manifest_file = backup_dir / "backup_manifest.json"
            if not manifest_file.exists():
                result.errors.append("Backup manifest not found")
                result.duration = time.time() - start_time
                return result
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Restore files and directories
            try:
                self._restore_from_backup(backup_path)
                result.steps_completed.append("restore_backup")
                
                # Update version
                backup_version = manifest.get('version', '0.1.0')
                with open(self.current_version_file, 'w') as f:
                    f.write(backup_version)
                self.current_version = backup_version
                
                result.success = True
                self.logger.info(f"Rollback to version {backup_version} completed")
                
            except Exception as e:
                error_msg = f"Failed to restore backup: {e}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"Rollback failed: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
        
        result.duration = time.time() - start_time
        return result
    
    def _restore_from_backup(self, backup_path: str):
        """Restore files from backup."""
        backup_dir = Path(backup_path)
        
        # Restore files
        for backup_file in backup_dir.glob("*"):
            if backup_file.is_file() and backup_file.name != "backup_manifest.json":
                target_file = self.config_dir / backup_file.name
                shutil.copy2(backup_file, target_file)
            
            elif backup_file.is_dir():
                target_dir = self.config_dir / backup_file.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(backup_file, target_dir)
    
    def _log_migration(self, from_version: str, to_version: str, result: MigrationResult):
        """Log migration details."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'from_version': from_version,
                'to_version': to_version,
                'success': result.success,
                'steps_completed': result.steps_completed,
                'steps_failed': result.steps_failed,
                'errors': result.errors,
                'duration': result.duration,
                'backup_created': result.backup_created
            }
            
            # Load existing log
            migration_log = []
            if self.migration_log_file.exists():
                with open(self.migration_log_file, 'r') as f:
                    migration_log = json.load(f)
            
            # Add new entry
            migration_log.append(log_entry)
            
            # Keep only last 10 migrations
            migration_log = migration_log[-10:]
            
            # Save log
            with open(self.migration_log_file, 'w') as f:
                json.dump(migration_log, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to log migration: {e}")
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            if self.migration_log_file.exists():
                with open(self.migration_log_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Failed to load migration history: {e}")
            return []
    
    def validate_migration(self, target_version: str = "0.2.0") -> Dict[str, Any]:
        """
        Validate that migration can be performed safely.
        
        Args:
            target_version: Target version to validate
            
        Returns:
            Validation result
        """
        validation = {
            'can_migrate': True,
            'warnings': [],
            'errors': [],
            'requirements_met': True
        }
        
        try:
            # Check if we're already at target version
            if not self.needs_migration(target_version):
                validation['warnings'].append("Already at target version")
                return validation
            
            # Check disk space
            try:
                import shutil
                total, used, free = shutil.disk_usage(self.config_dir)
                free_mb = free // (1024 * 1024)
                
                if free_mb < 100:  # Need at least 100MB free
                    validation['errors'].append(f"Insufficient disk space: {free_mb}MB available, need 100MB")
                    validation['can_migrate'] = False
            except:
                validation['warnings'].append("Could not check disk space")
            
            # Check backup directory
            if not self.backup_dir.exists():
                try:
                    self.backup_dir.mkdir(exist_ok=True)
                except Exception as e:
                    validation['errors'].append(f"Cannot create backup directory: {e}")
                    validation['can_migrate'] = False
            
            # Check migration steps
            pending = self.get_pending_migrations(target_version)
            if not pending:
                validation['warnings'].append("No migration steps found")
            
            # Check for critical files
            critical_files = ["setup.py"]
            for file_name in critical_files:
                if not (self.config_dir / file_name).exists():
                    validation['warnings'].append(f"Critical file missing: {file_name}")
            
            # Check Python version (for 0.2.0 requirement)
            import sys
            if sys.version_info < (3, 9):
                validation['errors'].append("Python 3.9+ required for version 0.2.0")
                validation['can_migrate'] = False
                validation['requirements_met'] = False
        
        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")
            validation['can_migrate'] = False
        
        return validation


# Command-line interface for migration
def migrate_from_command_line():
    """Command-line interface for migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIFrameworkRPC Migration Tool")
    parser.add_argument("--from-version", default="0.1.0", help="Source version")
    parser.add_argument("--to-version", default="0.2.0", help="Target version")
    parser.add_argument("--config-dir", default=".", help="Configuration directory")
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate migration")
    parser.add_argument("--rollback", help="Rollback to specified backup")
    parser.add_argument("--history", action="store_true", help="Show migration history")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize migration manager
    migration_manager = MigrationManager(args.config_dir, args.backup_dir)
    
    try:
        if args.history:
            history = migration_manager.get_migration_history()
            print("Migration History:")
            for entry in history:
                print(f"  {entry['timestamp']}: {entry['from_version']} -> {entry['to_version']} "
                      f"({'Success' if entry['success'] else 'Failed'})")
        
        elif args.rollback:
            print(f"Rolling back to backup: {args.rollback}")
            result = migration_manager.rollback(args.rollback)
            print(f"Rollback {'succeeded' if result.success else 'failed'}")
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
        
        elif args.validate_only:
            print(f"Validating migration to {args.to_version}...")
            validation = migration_manager.validate_migration(args.to_version)
            
            print(f"Can migrate: {validation['can_migrate']}")
            print(f"Requirements met: {validation['requirements_met']}")
            
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
            
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
        
        else:
            print(f"Migrating from {args.from_version} to {args.to_version}...")
            
            # Validate first
            validation = migration_manager.validate_migration(args.to_version)
            if not validation['can_migrate']:
                print("Migration cannot proceed:")
                for error in validation['errors']:
                    print(f"  Error: {error}")
                return 1
            
            # Perform migration
            result = migration_manager.migrate(
                args.to_version, 
                create_backup=not args.no_backup
            )
            
            print(f"Migration {'succeeded' if result.success else 'failed'}")
            print(f"Duration: {result.duration:.2f} seconds")
            
            if result.backup_created:
                print(f"Backup created: {result.backup_path}")
            
            if result.steps_completed:
                print("Completed steps:")
                for step in result.steps_completed:
                    print(f"  - {step}")
            
            if result.steps_failed:
                print("Failed steps:")
                for step in result.steps_failed:
                    print(f"  - {step}")
            
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
            
            return 0 if result.success else 1
    
    except KeyboardInterrupt:
        print("\nMigration cancelled by user")
        return 1
    except Exception as e:
        print(f"Migration error: {e}")
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(migrate_from_command_line())
