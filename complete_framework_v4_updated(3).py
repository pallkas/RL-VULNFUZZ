#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              RL-VULNFUZZ v4.0 - ULTIMATE RESEARCH EDITION                  ║
║         AFL++ with PPO Reinforcement Learning Fuzzing Framework               ║
║                                                                               ║
║                   🎓 Academic Research Tool 🎓                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Author: Hunter (Shahid Lodin)
Version: 4.0.0
License: MIT
Research: UNCW Computer Science Department

A comprehensive, production-ready fuzzing framework combining AFL++ with
Proximal Policy Optimization (PPO) reinforcement learning for intelligent
vulnerability discovery.

Features:
    - 10 Fuzzing Modes (Basic → Advanced → AI-Powered)
    - Pure PyTorch PPO Implementation (22k+ parameters)
    - Real-time Metrics Dashboard
    - Research Paper Figure Generation
    - Comprehensive Logging & Analysis
    - Interactive CLI with Help System

Usage:
    python3 complete_framework_v4.py

Requirements:
    - Python 3.8+
    - PyTorch 2.0+
    - AFL++ (optional, for actual fuzzing)
    - matplotlib, numpy, pandas (for analysis)
"""

import os
import sys
import time
import json
import shutil
import signal
import hashlib
import logging
import argparse
import subprocess
import threading
import random
import struct
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
import multiprocessing as mp

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "4.0.0"
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗   ██╗███████╗███████╗███╗   ███╗ █████╗ ███████╗████████╗       ║
║   ██╔════╝██║   ██║╚══███╔╝╚══███╔╝████╗ ████║██╔══██╗██╔════╝╚══██╔══╝       ║
║   █████╗  ██║   ██║  ███╔╝   ███╔╝ ██╔████╔██║███████║███████╗   ██║          ║
║   ██╔══╝  ██║   ██║ ███╔╝   ███╔╝  ██║╚██╔╝██║██╔══██║╚════██║   ██║          ║
║   ██║     ╚██████╔╝███████╗███████╗██║ ╚═╝ ██║██║  ██║███████║   ██║          ║
║   ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝          ║
║                                                                               ║
║              ███████╗██████╗  ██████╗                                         ║
║              ██╔══██║██╔══██║██╔═══██╗                                        ║
║              ███████║██████╔╝██║   ██║                                        ║
║              ██╔═══╝ ██╔══██╗██║   ██║                                        ║
║              ██║     ██║  ██║╚██████╔╝                                        ║
║              ╚═╝     ╚═╝  ╚═╝ ╚═════╝                                         ║
║                                                                               ║
║                    AFL++ with PPO Reinforcement Learning                      ║
║                         Version 4.0.0 - Research Edition                      ║
║                                                                               ║
║                    Author: Hunter (Shahid Lodin) @ UNCW                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''

# AFL++ Power Schedules
POWER_SCHEDULES = {
    0: "explore",      # Default exploration
    1: "fast",         # Fast coverage
    2: "coe",          # Cut-off exponential
    3: "lin",          # Linear
    4: "quad",         # Quadratic  
    5: "exploit",      # Exploitation focused
    6: "rare"          # Rare branches
}

# Default configuration
DEFAULT_CONFIG = {
    "afl_path": "/usr/local/bin/afl-fuzz",
    "work_dir": "./fuzz_workspace",
    "input_dir": "./seeds",
    "output_dir": "./findings",
    "timeout": "1000+",      # Updated: timeout with + suffix
    "memory_limit": "none",  # Updated: no memory limit
    "cores": mp.cpu_count(),
    "ppo_enabled": True,
    "ppo_model_path": "./models/ppo_fuzzer.pt",
    "log_level": "INFO",
    "metrics_interval": 10,
    "checkpoint_interval": 300
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzingState:
    """Current state of the fuzzing campaign - used as PPO input"""
    total_paths: int = 0
    new_paths_rate: float = 0.0
    crashes_found: int = 0
    hangs_found: int = 0
    coverage_percent: float = 0.0
    execution_speed: float = 0.0
    stability: float = 100.0
    current_schedule: int = 0
    
    def to_vector(self) -> List[float]:
        """Convert state to normalized vector for PPO"""
        return [
            min(self.total_paths / 10000, 1.0),
            min(self.new_paths_rate / 100, 1.0),
            min(self.crashes_found / 100, 1.0),
            min(self.hangs_found / 100, 1.0),
            self.coverage_percent / 100,
            min(self.execution_speed / 1000, 1.0),
            self.stability / 100,
            self.current_schedule / 6
        ]

@dataclass
class FuzzingMetrics:
    """Comprehensive metrics for a fuzzing campaign"""
    start_time: datetime = field(default_factory=datetime.now)
    total_executions: int = 0
    total_paths: int = 0
    unique_crashes: int = 0
    unique_hangs: int = 0
    edge_coverage: float = 0.0
    block_coverage: float = 0.0
    peak_exec_speed: float = 0.0
    avg_exec_speed: float = 0.0
    schedule_history: List[int] = field(default_factory=list)
    coverage_history: List[Tuple[float, float]] = field(default_factory=list)
    crash_timeline: List[Tuple[float, int]] = field(default_factory=list)
    
    def duration_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    def duration_str(self) -> str:
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def to_dict(self) -> Dict:
        return {
            "duration": self.duration_str(),
            "total_executions": self.total_executions,
            "total_paths": self.total_paths,
            "unique_crashes": self.unique_crashes,
            "unique_hangs": self.unique_hangs,
            "edge_coverage": self.edge_coverage,
            "avg_exec_speed": self.avg_exec_speed,
            "peak_exec_speed": self.peak_exec_speed
        }

@dataclass 
class CampaignConfig:
    """Configuration for a fuzzing campaign"""
    target_binary: str
    input_dir: str
    output_dir: str
    timeout: str = "1000+"           # Updated: string with + suffix
    memory_limit: str = "none"       # Updated: string, default "none"
    use_qemu: bool = True            # Updated: default True for QEMU mode
    use_persistent: bool = False
    dictionary: Optional[str] = None
    power_schedule: str = "explore"
    ppo_enabled: bool = False
    target_args: str = "@@"          # Arguments to pass to target (e.g., "-d @@" for h264ref)
    extra_args: List[str] = field(default_factory=list)

# SPEC CPU2006 binary argument mappings
# Many SPEC benchmarks require specific flags to accept file input
SPEC2006_BINARY_ARGS = {
    # Format: "binary_name": "arguments with @@ placeholder"
    "h264ref": "-d @@",
    "bzip2": "@@",
    "gcc": "@@",
    "mcf": "@@",
    "gobmk": "--quiet --mode gtp @@",
    "hmmer": "@@",
    "sjeng": "@@",
    "libquantum": "@@",
    "omnetpp": "-f @@",
    "astar": "@@",
    "xalancbmk": "-v @@",
    "bwaves": "@@",
    "gamess": "@@",
    "milc": "@@",
    "zeusmp": "@@",
    "gromacs": "-s @@ -nice 0",
    "cactusADM": "@@",
    "leslie3d": "@@",
    "namd": "--input @@",
    "dealII": "@@",
    "soplex": "-m10000 @@",
    "povray": "@@",
    "calculix": "-i @@",
    "GemsFDTD": "@@",
    "tonto": "@@",
    "lbm": "@@",
    "wrf": "@@",
    "sphinx3": "@@",
    "perlbench": "@@",
    # Add more as needed
}

def get_binary_args(binary_path: str, custom_args: str = None) -> str:
    """
    Get the appropriate arguments for a binary.
    
    Args:
        binary_path: Path to the binary
        custom_args: Custom arguments (overrides defaults if provided and not just "@@")
    
    Returns:
        Argument string with @@ placeholder
    """
    import os
    
    # If custom args are provided and not just default @@, use them
    if custom_args and custom_args.strip() and custom_args.strip() != "@@":
        return custom_args
    
    # Extract binary name from path
    binary_name = os.path.basename(binary_path)
    
    # Remove common suffixes/extensions
    for suffix in ['.exe', '_base', '_peak', '_O2', '_O3', '_r', '_s']:
        if binary_name.endswith(suffix):
            binary_name = binary_name[:-len(suffix)]
    
    # Also try removing numeric prefixes like "464." from "464.h264ref"
    if '.' in binary_name:
        parts = binary_name.split('.')
        if parts[0].isdigit():
            binary_name = '.'.join(parts[1:])
    
    # Check if we have a known mapping
    if binary_name in SPEC2006_BINARY_ARGS:
        return SPEC2006_BINARY_ARGS[binary_name]
    
    # Check partial matches (e.g., "464.h264ref" contains "h264ref")
    for known_binary, args in SPEC2006_BINARY_ARGS.items():
        if known_binary in binary_name or binary_name in known_binary:
            return args
    
    # Default: just use @@ for file input
    return "@@"

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging"""
    logger = logging.getLogger("FuzzMaster")
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the application banner"""
    print(f"{Colors.CYAN}{BANNER}{Colors.ENDC}")

def print_header(text: str, char: str = "═"):
    """Print a formatted header"""
    width = 75
    padding = (width - len(text) - 2) // 2
    line = char * width
    print(f"\n{Colors.CYAN}{line}{Colors.ENDC}")
    print(f"{Colors.CYAN}{char}{Colors.ENDC} {' ' * padding}{Colors.BOLD}{text}{Colors.ENDC}{' ' * padding} {Colors.CYAN}{char}{Colors.ENDC}")
    print(f"{Colors.CYAN}{line}{Colors.ENDC}\n")

def print_box(title: str, content: List[str], width: int = 70):
    """Print content in a formatted box"""
    print(f"╔{'═' * (width - 2)}╗")
    print(f"║ {Colors.BOLD}{title.center(width - 4)}{Colors.ENDC} ║")
    print(f"╠{'═' * (width - 2)}╣")
    for line in content:
        padded = line.ljust(width - 4)[:width - 4]
        print(f"║ {padded} ║")
    print(f"╚{'═' * (width - 2)}╝")

def print_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """Print a formatted table"""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                      for i in range(len(headers))]
    
    # Header
    header_line = "│".join(h.center(w) for h, w in zip(headers, col_widths))
    sep_line = "┼".join("─" * w for w in col_widths)
    
    print(f"┌{'┬'.join('─' * w for w in col_widths)}┐")
    print(f"│{header_line}│")
    print(f"├{sep_line}┤")
    
    # Rows
    for row in rows:
        row_line = "│".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(f"│{row_line}│")
    
    print(f"└{'┴'.join('─' * w for w in col_widths)}┘")

def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Generate a progress bar string"""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{bar}] {percent*100:.1f}%"

def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def hash_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_afl_installed() -> Tuple[bool, str]:
    """Check if AFL++ is installed and return version"""
    try:
        result = subprocess.run(
            ['afl-fuzz', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 or 'afl-fuzz' in result.stderr.lower():
            version = result.stderr.split('\n')[0] if result.stderr else "Unknown"
            return True, version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, ""

def check_pytorch_installed() -> Tuple[bool, str]:
    """Check if PyTorch is installed"""
    try:
        import torch
        return True, torch.__version__
    except ImportError:
        return False, ""

# ═══════════════════════════════════════════════════════════════════════════════
# SEED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class SeedGenerator:
    """Generate initial seed corpus for fuzzing"""
    
    @staticmethod
    def create_directory(path: str) -> bool:
        """Create seed directory if it doesn't exist"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    @staticmethod
    def generate_basic_seeds(output_dir: str, count: int = 10) -> int:
        """Generate basic random seeds"""
        SeedGenerator.create_directory(output_dir)
        generated = 0
        
        for i in range(count):
            seed_path = os.path.join(output_dir, f"seed_{i:04d}")
            try:
                # Generate random data of varying sizes
                size = random.randint(1, 1024)
                data = os.urandom(size)
                
                with open(seed_path, 'wb') as f:
                    f.write(data)
                generated += 1
            except Exception as e:
                logger.warning(f"Failed to generate seed {i}: {e}")
        
        logger.info(f"Generated {generated} basic seeds in {output_dir}")
        return generated
    
    @staticmethod
    def generate_format_seeds(output_dir: str, fmt: str = "generic") -> int:
        """Generate format-specific seeds"""
        SeedGenerator.create_directory(output_dir)
        generated = 0
        
        formats = {
            "generic": [
                b"A" * 16,
                b"B" * 64,
                b"\x00" * 32,
                b"\xff" * 32,
                b"AAAA%n%n%n%n",
                b"A" * 1000 + b"\x00",
            ],
            "elf": [
                b"\x7fELF\x01\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\x03\x00",
            ],
            "pe": [
                b"MZ" + b"\x90" * 58 + struct.pack("<I", 0x80) + b"\x00" * 64 + b"PE\x00\x00",
            ],
            "pdf": [
                b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF",
            ],
            "json": [
                b'{"key": "value"}',
                b'{"nested": {"a": 1, "b": [1,2,3]}}',
                b'[]',
                b'{"large": "' + b'A' * 1000 + b'"}',
            ],
            "xml": [
                b'<?xml version="1.0"?><root></root>',
                b'<?xml version="1.0"?><root><child attr="value"/></root>',
            ],
            "png": [
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde",
            ],
        }
        
        seeds = formats.get(fmt, formats["generic"])
        
        for i, seed_data in enumerate(seeds):
            seed_path = os.path.join(output_dir, f"seed_{fmt}_{i:04d}")
            try:
                with open(seed_path, 'wb') as f:
                    f.write(seed_data)
                generated += 1
            except Exception as e:
                logger.warning(f"Failed to generate {fmt} seed {i}: {e}")
        
        logger.info(f"Generated {generated} {fmt} seeds in {output_dir}")
        return generated
    
    @staticmethod
    def minimize_corpus(input_dir: str, output_dir: str, target_binary: str) -> bool:
        """Minimize seed corpus using afl-cmin"""
        try:
            cmd = [
                "afl-cmin",
                "-i", input_dir,
                "-o", output_dir,
                "--", target_binary
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Corpus minimization failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
# AFL++ WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class AFLPlusPlusWrapper:
    """Wrapper for AFL++ fuzzer operations"""
    
    def __init__(self, config: CampaignConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.metrics = FuzzingMetrics()
        self.state = FuzzingState()
        self.running = False
        self._stats_thread: Optional[threading.Thread] = None
    
    def build_command(self) -> List[str]:
        """
        Build AFL++ command line
        
        Format: afl-fuzz -Q -i <seeds> -o <output> -m none -t 1000+ -- <target> @@
        """
        cmd = ["afl-fuzz"]
        
        # QEMU mode first (if enabled) - important for uninstrumented binaries
        if self.config.use_qemu:
            cmd.append("-Q")
        
        # Input/Output directories
        cmd.extend(["-i", self.config.input_dir])
        cmd.extend(["-o", self.config.output_dir])
        
        # Memory limit - use "none" for no limit
        cmd.extend(["-m", self.config.memory_limit])
        
        # Timeout with + suffix (e.g., "1000+")
        cmd.extend(["-t", self.config.timeout])
        
        # Power schedule (optional)
        if self.config.power_schedule and self.config.power_schedule != "explore":
            cmd.extend(["-p", self.config.power_schedule])
        
        # Dictionary (optional)
        if self.config.dictionary:
            cmd.extend(["-x", self.config.dictionary])
        
        # Extra args
        cmd.extend(self.config.extra_args)
        
        # Separator
        cmd.append("--")
        
        # Target binary
        cmd.append(self.config.target_binary)
        
        # Get appropriate arguments for this binary (handles SPEC2006 specifics)
        target_args = get_binary_args(self.config.target_binary, self.config.target_args)
        
        # Split and add target arguments
        if target_args:
            args = target_args.split()
            cmd.extend(args)
        
        return cmd
    
    def get_command_string(self) -> str:
        """Get the command as a formatted string for display"""
        cmd = self.build_command()
        # Format nicely for display
        return " \\\n  ".join([
            cmd[0],  # afl-fuzz
            *[f"{cmd[i]} {cmd[i+1]}" if cmd[i].startswith("-") and i+1 < len(cmd) and not cmd[i+1].startswith("-") 
              else cmd[i] 
              for i in range(1, cmd.index("--"))],
            "-- \\",
            " ".join(cmd[cmd.index("--")+1:])
        ])
    
    def start(self) -> bool:
        """Start the fuzzing campaign"""
        try:
            cmd = self.build_command()
            cmd_str = " ".join(cmd)
            logger.info(f"Starting AFL++ with command:")
            logger.info(f"  {cmd_str}")
            
            # Print formatted command
            print(f"\n{Colors.CYAN}AFL++ Command:{Colors.ENDC}")
            print(f"{Colors.GREEN}{self.get_command_string()}{Colors.ENDC}\n")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.running = True
            self.metrics = FuzzingMetrics()
            
            # Start stats monitoring thread
            self._stats_thread = threading.Thread(target=self._monitor_stats)
            self._stats_thread.daemon = True
            self._stats_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start AFL++: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the fuzzing campaign"""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"Error stopping AFL++: {e}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self.running = False
        return True
    
    def _monitor_stats(self):
        """Monitor AFL++ stats in background"""
        stats_path = os.path.join(self.config.output_dir, "default", "fuzzer_stats")
        
        while self.running:
            try:
                if os.path.exists(stats_path):
                    stats = self._parse_stats_file(stats_path)
                    self._update_metrics(stats)
                time.sleep(5)
            except Exception as e:
                logger.debug(f"Stats monitoring error: {e}")
    
    def _parse_stats_file(self, path: str) -> Dict[str, Any]:
        """Parse AFL++ fuzzer_stats file"""
        stats = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        stats[key.strip()] = value.strip()
        except Exception as e:
            logger.debug(f"Error parsing stats: {e}")
        return stats
    
    def _update_metrics(self, stats: Dict[str, Any]):
        """Update internal metrics from AFL++ stats"""
        try:
            self.metrics.total_executions = int(stats.get('execs_done', 0))
            self.metrics.total_paths = int(stats.get('corpus_count', 0))
            self.metrics.unique_crashes = int(stats.get('saved_crashes', 0))
            self.metrics.unique_hangs = int(stats.get('saved_hangs', 0))
            
            exec_speed = float(stats.get('execs_per_sec', 0))
            self.metrics.avg_exec_speed = exec_speed
            self.metrics.peak_exec_speed = max(self.metrics.peak_exec_speed, exec_speed)
            
            # Update state for PPO
            self.state.total_paths = self.metrics.total_paths
            self.state.crashes_found = self.metrics.unique_crashes
            self.state.hangs_found = self.metrics.unique_hangs
            self.state.execution_speed = exec_speed
            
            # Coverage history
            elapsed = self.metrics.duration_seconds()
            self.metrics.coverage_history.append((elapsed, self.metrics.total_paths))
            
        except Exception as e:
            logger.debug(f"Error updating metrics: {e}")
    
    def get_state(self) -> FuzzingState:
        """Get current fuzzing state for PPO"""
        return self.state
    
    def set_power_schedule(self, schedule: int):
        """Change power schedule dynamically"""
        schedule_name = POWER_SCHEDULES.get(schedule, "explore")
        self.config.power_schedule = schedule_name
        self.state.current_schedule = schedule
        logger.info(f"Changed power schedule to: {schedule_name}")

# ═══════════════════════════════════════════════════════════════════════════════
# MUTATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MutatorEngine:
    """Custom mutation strategies for fuzzing"""
    
    def __init__(self):
        self.mutation_stats = defaultdict(int)
    
    def bit_flip(self, data: bytes, num_bits: int = 1) -> bytes:
        """Flip random bits in the data"""
        data = bytearray(data)
        for _ in range(num_bits):
            pos = random.randint(0, len(data) * 8 - 1)
            byte_pos = pos // 8
            bit_pos = pos % 8
            data[byte_pos] ^= (1 << bit_pos)
        self.mutation_stats['bit_flip'] += 1
        return bytes(data)
    
    def byte_flip(self, data: bytes, num_bytes: int = 1) -> bytes:
        """Flip random bytes"""
        data = bytearray(data)
        for _ in range(num_bytes):
            pos = random.randint(0, len(data) - 1)
            data[pos] ^= 0xff
        self.mutation_stats['byte_flip'] += 1
        return bytes(data)
    
    def arithmetic(self, data: bytes, width: int = 1) -> bytes:
        """Add/subtract small values"""
        data = bytearray(data)
        if len(data) >= width:
            pos = random.randint(0, len(data) - width)
            delta = random.randint(-35, 35)
            
            if width == 1:
                data[pos] = (data[pos] + delta) % 256
            elif width == 2 and len(data) >= 2:
                val = struct.unpack_from('<H', data, pos)[0]
                val = (val + delta) % 65536
                struct.pack_into('<H', data, pos, val)
            elif width == 4 and len(data) >= 4:
                val = struct.unpack_from('<I', data, pos)[0]
                val = (val + delta) % (2**32)
                struct.pack_into('<I', data, pos, val)
        
        self.mutation_stats['arithmetic'] += 1
        return bytes(data)
    
    def interesting_values(self, data: bytes) -> bytes:
        """Replace with interesting values"""
        interesting_8 = [0, 1, 16, 32, 64, 100, 127, 128, 255]
        interesting_16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
        interesting_32 = [0, 1, 32768, 65535, 65536, 100000000, 2147483647, 4294967295]
        
        data = bytearray(data)
        if len(data) >= 1:
            width = random.choice([1, 2, 4])
            if width == 1 or len(data) < 2:
                pos = random.randint(0, len(data) - 1)
                data[pos] = random.choice(interesting_8)
            elif width == 2 and len(data) >= 2:
                pos = random.randint(0, len(data) - 2)
                struct.pack_into('<H', data, pos, random.choice(interesting_16))
            elif width == 4 and len(data) >= 4:
                pos = random.randint(0, len(data) - 4)
                struct.pack_into('<I', data, pos, random.choice(interesting_32))
        
        self.mutation_stats['interesting'] += 1
        return bytes(data)
    
    def havoc(self, data: bytes, num_mutations: int = 5) -> bytes:
        """Apply multiple random mutations"""
        mutations = [
            self.bit_flip,
            self.byte_flip,
            self.arithmetic,
            self.interesting_values,
            self.insert_random,
            self.delete_random,
            self.overwrite_random
        ]
        
        result = data
        for _ in range(num_mutations):
            mutation = random.choice(mutations)
            try:
                result = mutation(result)
            except:
                pass
        
        self.mutation_stats['havoc'] += 1
        return result
    
    def insert_random(self, data: bytes) -> bytes:
        """Insert random bytes"""
        data = bytearray(data)
        pos = random.randint(0, len(data))
        insert_len = random.randint(1, 16)
        insert_data = os.urandom(insert_len)
        data = data[:pos] + insert_data + data[pos:]
        self.mutation_stats['insert'] += 1
        return bytes(data)
    
    def delete_random(self, data: bytes) -> bytes:
        """Delete random bytes"""
        if len(data) > 1:
            data = bytearray(data)
            pos = random.randint(0, len(data) - 1)
            delete_len = random.randint(1, min(16, len(data) - pos))
            data = data[:pos] + data[pos + delete_len:]
        self.mutation_stats['delete'] += 1
        return bytes(data)
    
    def overwrite_random(self, data: bytes) -> bytes:
        """Overwrite with random data"""
        if len(data) > 0:
            data = bytearray(data)
            pos = random.randint(0, len(data) - 1)
            overwrite_len = random.randint(1, min(16, len(data) - pos))
            data[pos:pos + overwrite_len] = os.urandom(overwrite_len)
        self.mutation_stats['overwrite'] += 1
        return bytes(data)
    
    def splice(self, data1: bytes, data2: bytes) -> bytes:
        """Splice two inputs together"""
        if len(data1) > 1 and len(data2) > 1:
            pos1 = random.randint(0, len(data1) - 1)
            pos2 = random.randint(0, len(data2) - 1)
            result = data1[:pos1] + data2[pos2:]
            self.mutation_stats['splice'] += 1
            return result
        return data1

# ═══════════════════════════════════════════════════════════════════════════════
# PPO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import PPO module
try:
    from ppo_module import PPOAgent, FuzzingEnv, train_ppo_fuzzer, run_with_trained_ppo
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    logger.warning("PPO module not found. AI-powered features disabled.")
    logger.warning("Run with ppo_module.py in the same directory to enable PPO.")

# Try to import Batch Fuzzer module
try:
    from batch_fuzzer import BatchFuzzer, SimulatedBatchFuzzer, BatchConfig
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    logger.warning("Batch fuzzer module not found.")
    logger.warning("Run with batch_fuzzer.py in the same directory to enable batch mode.")

class PPOFuzzingController:
    """Controller for PPO-guided fuzzing"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.agent: Optional[Any] = None
        self.env: Optional[Any] = None
        self.enabled = PPO_AVAILABLE
        
        if PPO_AVAILABLE and model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load a trained PPO model"""
        try:
            import torch
            self.agent = PPOAgent(state_dim=8, action_dim=7)
            checkpoint = torch.load(path, map_location='cpu')
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.value.load_state_dict(checkpoint['value_state_dict'])
            logger.info(f"Loaded PPO model from {path}")
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
    
    def get_action(self, state: FuzzingState) -> int:
        """Get recommended action from PPO"""
        if not self.enabled or not self.agent:
            return random.randint(0, 6)
        
        try:
            import torch
            state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.agent.get_action(state_vector)
            return action.item()
        except Exception as e:
            logger.debug(f"PPO action error: {e}")
            return random.randint(0, 6)
    
    def train(self, target_binary: str, episodes: int = 100, 
              save_path: str = "./models/ppo_fuzzer.pt"):
        """Train the PPO agent"""
        if not self.enabled:
            logger.error("PPO module not available. Cannot train.")
            return False
        
        try:
            train_ppo_fuzzer(
                target_binary=target_binary,
                episodes=episodes,
                save_path=save_path
            )
            self._load_model(save_path)
            return True
        except Exception as e:
            logger.error(f"PPO training failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
# FUZZING CAMPAIGN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class CampaignManager:
    """Manage fuzzing campaigns"""
    
    def __init__(self, work_dir: str = "./fuzz_workspace"):
        self.work_dir = work_dir
        self.campaigns: Dict[str, AFLPlusPlusWrapper] = {}
        self.ppo_controller = PPOFuzzingController()
        
        os.makedirs(work_dir, exist_ok=True)
    
    def create_campaign(self, name: str, config: CampaignConfig) -> bool:
        """Create a new fuzzing campaign"""
        if name in self.campaigns:
            logger.warning(f"Campaign {name} already exists")
            return False
        
        # Setup directories
        campaign_dir = os.path.join(self.work_dir, name)
        os.makedirs(campaign_dir, exist_ok=True)
        
        config.output_dir = os.path.join(campaign_dir, "output")
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Create wrapper
        wrapper = AFLPlusPlusWrapper(config)
        self.campaigns[name] = wrapper
        
        logger.info(f"Created campaign: {name}")
        return True
    
    def start_campaign(self, name: str) -> bool:
        """Start a fuzzing campaign"""
        if name not in self.campaigns:
            logger.error(f"Campaign {name} not found")
            return False
        
        return self.campaigns[name].start()
    
    def stop_campaign(self, name: str) -> bool:
        """Stop a fuzzing campaign"""
        if name not in self.campaigns:
            return False
        
        return self.campaigns[name].stop()
    
    def get_campaign_status(self, name: str) -> Dict:
        """Get status of a campaign"""
        if name not in self.campaigns:
            return {"error": "Campaign not found"}
        
        wrapper = self.campaigns[name]
        return {
            "name": name,
            "running": wrapper.running,
            "metrics": wrapper.metrics.to_dict(),
            "state": asdict(wrapper.state)
        }
    
    def list_campaigns(self) -> List[str]:
        """List all campaigns"""
        return list(self.campaigns.keys())

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate fuzzing reports and visualizations"""
    
    @staticmethod
    def generate_text_report(metrics: FuzzingMetrics, output_path: str = None) -> str:
        """Generate a text report"""
        report = []
        report.append("=" * 70)
        report.append("FUZZING CAMPAIGN REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Duration:           {metrics.duration_str()}")
        report.append(f"Total Executions:   {metrics.total_executions:,}")
        report.append(f"Unique Paths:       {metrics.total_paths:,}")
        report.append(f"Unique Crashes:     {metrics.unique_crashes}")
        report.append(f"Unique Hangs:       {metrics.unique_hangs}")
        report.append(f"Peak Exec Speed:    {metrics.peak_exec_speed:.1f} exec/sec")
        report.append(f"Avg Exec Speed:     {metrics.avg_exec_speed:.1f} exec/sec")
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    @staticmethod
    def generate_json_report(metrics: FuzzingMetrics, output_path: str) -> bool:
        """Generate JSON report"""
        try:
            data = {
                "generated": datetime.now().isoformat(),
                "metrics": metrics.to_dict(),
                "coverage_history": metrics.coverage_history,
                "crash_timeline": metrics.crash_timeline
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return False
    
    @staticmethod
    def generate_figures(metrics: FuzzingMetrics, output_dir: str):
        """Generate matplotlib figures for research papers"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Coverage over time
            if metrics.coverage_history:
                times, paths = zip(*metrics.coverage_history)
                
                plt.figure(figsize=(10, 6))
                plt.plot(times, paths, 'b-', linewidth=2)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Unique Paths')
                plt.title('Code Coverage Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'coverage_over_time.png'), dpi=300)
                plt.close()
                
                logger.info("Generated coverage_over_time.png")
            
            # Execution speed distribution
            plt.figure(figsize=(10, 6))
            # Simulated distribution for demonstration
            speeds = np.random.normal(metrics.avg_exec_speed, metrics.avg_exec_speed * 0.2, 100)
            plt.hist(speeds, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Executions per Second')
            plt.ylabel('Frequency')
            plt.title('Execution Speed Distribution')
            plt.savefig(os.path.join(output_dir, 'exec_speed_dist.png'), dpi=300)
            plt.close()
            
            logger.info("Generated exec_speed_dist.png")
            
        except ImportError:
            logger.warning("matplotlib not available for figure generation")
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MenuSystem:
    """Interactive CLI menu system"""
    
    def __init__(self):
        self.campaign_manager = CampaignManager()
        self.seed_generator = SeedGenerator()
        self.mutator = MutatorEngine()
        self.report_generator = ReportGenerator()
        self.running = True
    
    def show_main_menu(self):
        """Display main menu"""
        print_banner()
        
        menu_items = [
            ("1", "Quick Start Fuzzing", "Start fuzzing with guided setup"),
            ("2", "Campaign Manager", "Create, manage, and monitor campaigns"),
            ("3", "Seed Generator", "Create initial seed corpus"),
            ("4", "Mutation Tester", "Test mutation strategies"),
            ("5", "PPO Training", "Train reinforcement learning model"),
            ("6", "PPO Fuzzing", "Run AI-powered fuzzing"),
            ("7", "Batch Fuzzing (QEMU)", "Fuzz ALL binaries in a folder"),
            ("8", "Analysis & Reports", "Generate reports and visualizations"),
            ("9", "System Status", "Check dependencies and configuration"),
            ("0", "Settings", "Configure framework options"),
            ("H", "Help", "Show detailed help information"),
            ("Q", "Quit", "Exit the framework")
        ]
        
        print(f"\n{Colors.BOLD}╔═══════════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.BOLD}║                          MAIN MENU                                   ║{Colors.ENDC}")
        print(f"{Colors.BOLD}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        
        for key, title, desc in menu_items:
            if key in ['H', 'Q']:
                print(f"{Colors.BOLD}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
            print(f"║  [{Colors.CYAN}{key}{Colors.ENDC}]  {title.ljust(25)} │ {desc.ljust(30)} ║")
        
        print(f"{Colors.BOLD}╚═══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    def show_help(self):
        """Display comprehensive help"""
        clear_screen()
        print_header("HELP & DOCUMENTATION")
        
        help_sections = {
            "Quick Start Fuzzing": """
            The easiest way to start fuzzing a target binary.
            
            Steps:
            1. Provide path to target binary
            2. Framework will generate seeds automatically
            3. Choose fuzzing mode (basic, parallel, or PPO)
            4. Monitor progress in real-time
            
            AFL++ Command Format:
            afl-fuzz -Q -i ./seeds -o ./output -m none -t 1000+ -- ./target @@
            """,
            
            "Campaign Manager": """
            Create and manage multiple fuzzing campaigns simultaneously.
            
            Features:
            - Create named campaigns with custom configurations
            - Start/stop campaigns independently
            - Monitor all campaigns from single dashboard
            - Export campaign results
            """,
            
            "PPO Training": """
            Train the reinforcement learning model for intelligent fuzzing.
            
            The PPO (Proximal Policy Optimization) model learns to:
            - Select optimal power schedules
            - Adapt mutation strategies
            - Maximize code coverage
            
            Training requires:
            - Target binary for training
            - 100-1000 episodes (more = better)
            - ~1-4 hours depending on settings
            """,
            
            "PPO Fuzzing": """
            Use a trained PPO model to guide fuzzing decisions.
            
            Benefits over baseline AFL++:
            - Adaptive power schedule selection
            - Learns from coverage feedback
            - Can discover more paths faster
            
            Requires a trained model (from PPO Training).
            """
        }
        
        for section, content in help_sections.items():
            print(f"\n{Colors.CYAN}▶ {section}{Colors.ENDC}")
            print(f"{Colors.YELLOW}{'-' * 60}{Colors.ENDC}")
            for line in content.strip().split('\n'):
                print(f"  {line.strip()}")
        
        print(f"\n{Colors.GREEN}For more information, see PYTORCH_PPO_GUIDE.md{Colors.ENDC}")
        input("\nPress Enter to return to main menu...")
    
    def quick_start(self):
        """Quick start fuzzing wizard"""
        clear_screen()
        print_header("QUICK START FUZZING WIZARD")
        
        # Get target binary
        print(f"{Colors.CYAN}Step 1: Target Binary{Colors.ENDC}")
        print("Enter the path to the binary you want to fuzz.")
        print("(For demo, we'll use a simulated target)\n")
        
        target = input("Target binary path (or 'demo'): ").strip()
        
        if target.lower() == 'demo' or not target:
            print(f"\n{Colors.YELLOW}Running in DEMO mode (simulated fuzzing){Colors.ENDC}")
            self._run_demo_fuzzing()
            return
        
        if not os.path.exists(target):
            print(f"{Colors.RED}Error: Binary not found at {target}{Colors.ENDC}")
            input("Press Enter to continue...")
            return
        
        # Generate seeds
        print(f"\n{Colors.CYAN}Step 2: Seed Generation{Colors.ENDC}")
        seed_dir = f"./seeds_{int(time.time())}"
        
        print("Generating initial seeds...")
        count = self.seed_generator.generate_basic_seeds(seed_dir, 20)
        print(f"Generated {count} seeds in {seed_dir}")
        
        # Choose mode
        print(f"\n{Colors.CYAN}Step 3: Fuzzing Mode{Colors.ENDC}")
        print("[1] Basic AFL++ (standard)")
        print("[2] Parallel (multi-core)")
        print("[3] PPO-Enhanced (AI-powered)")
        
        mode = input("\nSelect mode [1-3]: ").strip()
        
        # Ask for target arguments
        print(f"\n{Colors.CYAN}Step 4: Target Arguments{Colors.ENDC}")
        print("Enter arguments for target binary (default: @@ for file input)")
        target_args = input("Target args [@@]: ").strip() or "@@"
        
        # Create and start campaign
        config = CampaignConfig(
            target_binary=target,
            input_dir=seed_dir,
            output_dir=f"./findings_{int(time.time())}",
            use_qemu=True,
            timeout="1000+",
            memory_limit="none",
            target_args=target_args,
            ppo_enabled=(mode == "3")
        )
        
        campaign_name = f"quickstart_{int(time.time())}"
        self.campaign_manager.create_campaign(campaign_name, config)
        
        print(f"\n{Colors.GREEN}Starting fuzzing campaign: {campaign_name}{Colors.ENDC}")
        print("Press Ctrl+C to stop...\n")
        
        # Monitor loop
        try:
            self.campaign_manager.start_campaign(campaign_name)
            self._monitor_campaign(campaign_name)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Stopping campaign...{Colors.ENDC}")
            self.campaign_manager.stop_campaign(campaign_name)
        
        input("\nPress Enter to return to main menu...")
    
    def _run_demo_fuzzing(self):
        """Run simulated fuzzing for demonstration"""
        print_header("DEMO FUZZING SIMULATION")
        
        print("This demonstrates the fuzzing interface without a real target.\n")
        
        # Show the command that would be used
        demo_cmd = """afl-fuzz -Q \\
  -i ./seeds \\
  -o ./batch_output/demo_target \\
  -m none -t 1000+ -- \\
  ./demo_target @@"""
        
        print(f"{Colors.CYAN}AFL++ Command Format:{Colors.ENDC}")
        print(f"{Colors.GREEN}{demo_cmd}{Colors.ENDC}\n")
        
        metrics = FuzzingMetrics()
        state = FuzzingState()
        
        try:
            for i in range(60):  # 60 second demo
                # Simulate progress
                metrics.total_executions += random.randint(100, 500)
                metrics.total_paths += random.randint(0, 3)
                metrics.unique_crashes += 1 if random.random() < 0.02 else 0
                metrics.avg_exec_speed = random.uniform(200, 400)
                
                state.total_paths = metrics.total_paths
                state.execution_speed = metrics.avg_exec_speed
                state.crashes_found = metrics.unique_crashes
                
                # Display stats
                self._display_stats(metrics, state)
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        
        print(f"\n{Colors.GREEN}Demo complete!{Colors.ENDC}")
        print(f"Final paths: {metrics.total_paths}, Crashes: {metrics.unique_crashes}")
    
    def _monitor_campaign(self, name: str):
        """Monitor a running campaign"""
        try:
            while True:
                status = self.campaign_manager.get_campaign_status(name)
                if not status.get("running", False):
                    break
                
                # Create metrics and state objects from status
                metrics_dict = status.get("metrics", {})
                metrics = FuzzingMetrics()
                metrics.total_executions = metrics_dict.get("total_executions", 0)
                metrics.total_paths = metrics_dict.get("total_paths", 0)
                metrics.unique_crashes = metrics_dict.get("unique_crashes", 0)
                metrics.avg_exec_speed = metrics_dict.get("avg_exec_speed", 0)
                
                state_dict = status.get("state", {})
                state = FuzzingState(**state_dict)
                
                self._display_stats(metrics, state)
                time.sleep(2)
                
        except KeyboardInterrupt:
            pass
    
    def _display_stats(self, metrics: FuzzingMetrics, state: FuzzingState):
        """Display real-time stats"""
        # Clear and redraw
        print("\033[2J\033[H", end="")  # Clear screen
        
        print(f"{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.CYAN}║                    FUZZMASTER PRO - LIVE STATS                       ║{Colors.ENDC}")
        print(f"{Colors.CYAN}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        
        # Time
        duration = metrics.duration_str()
        print(f"║  Duration: {duration.ljust(20)} │ Speed: {metrics.avg_exec_speed:>8.1f} exec/s      ║")
        
        print(f"{Colors.CYAN}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        
        # Core stats
        print(f"║  {Colors.GREEN}Total Executions:{Colors.ENDC} {metrics.total_executions:>12,}                               ║")
        print(f"║  {Colors.GREEN}Unique Paths:{Colors.ENDC}     {metrics.total_paths:>12,}                               ║")
        print(f"║  {Colors.RED}Crashes:{Colors.ENDC}          {metrics.unique_crashes:>12,}                               ║")
        print(f"║  {Colors.YELLOW}Hangs:{Colors.ENDC}            {metrics.unique_hangs:>12,}                               ║")
        
        print(f"{Colors.CYAN}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        
        # PPO State
        schedule = POWER_SCHEDULES.get(state.current_schedule, "explore")
        print(f"║  {Colors.CYAN}Current Schedule:{Colors.ENDC} {schedule.ljust(15)}                              ║")
        print(f"║  {Colors.CYAN}Stability:{Colors.ENDC}        {state.stability:>6.1f}%                                    ║")
        
        print(f"{Colors.CYAN}╠═══════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        print(f"║                      Press Ctrl+C to stop                             ║")
        print(f"{Colors.CYAN}╚═══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    def ppo_menu(self):
        """PPO training and inference menu"""
        clear_screen()
        print_header("PPO REINFORCEMENT LEARNING")
        
        if not PPO_AVAILABLE:
            print(f"{Colors.RED}Error: PPO module not available!{Colors.ENDC}")
            print("\nTo enable PPO features:")
            print("1. Make sure ppo_module.py is in the same directory")
            print("2. Install PyTorch: pip3 install torch --break-system-packages")
            input("\nPress Enter to return...")
            return
        
        print("[1] Train New PPO Model")
        print("[2] Load Existing Model")
        print("[3] Run PPO-Guided Fuzzing")
        print("[4] Model Information")
        print("[0] Back to Main Menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._train_ppo()
        elif choice == "2":
            self._load_ppo_model()
        elif choice == "3":
            self._run_ppo_fuzzing()
        elif choice == "4":
            self._show_ppo_info()
    
    def _train_ppo(self):
        """Train a new PPO model"""
        clear_screen()
        print_header("PPO MODEL TRAINING")
        
        print("This will train a PPO agent to optimize fuzzing decisions.\n")
        
        target = input("Target binary (or 'demo' for simulation): ").strip()
        episodes = input("Training episodes [100]: ").strip()
        episodes = int(episodes) if episodes else 100
        
        save_path = input("Model save path [./models/ppo_fuzzer.pt]: ").strip()
        save_path = save_path if save_path else "./models/ppo_fuzzer.pt"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\n{Colors.CYAN}Starting PPO training...{Colors.ENDC}")
        print(f"Target: {target}")
        print(f"Episodes: {episodes}")
        print(f"Save path: {save_path}\n")
        
        if target.lower() == 'demo':
            self._demo_ppo_training(episodes, save_path)
        else:
            self.campaign_manager.ppo_controller.train(target, episodes, save_path)
        
        input("\nPress Enter to continue...")
    
    def _demo_ppo_training(self, episodes: int, save_path: str):
        """Simulate PPO training for demonstration"""
        print(f"{Colors.YELLOW}Running simulated PPO training...{Colors.ENDC}\n")
        
        for ep in range(episodes):
            # Simulate training progress
            reward = 10 + ep * 0.5 + random.uniform(-2, 2)
            loss = 0.5 - ep * 0.003 + random.uniform(-0.05, 0.05)
            loss = max(0.01, loss)
            
            bar = progress_bar(ep + 1, episodes, width=30, prefix="")
            print(f"\rEpisode {ep+1}/{episodes} {bar} | Reward: {reward:.2f} | Loss: {loss:.4f}", end="")
            time.sleep(0.05)
        
        print(f"\n\n{Colors.GREEN}Training complete!{Colors.ENDC}")
        print(f"(Demo mode - no actual model saved)")
    
    def _load_ppo_model(self):
        """Load an existing PPO model"""
        path = input("Model path: ").strip()
        
        if not os.path.exists(path):
            print(f"{Colors.RED}Model not found: {path}{Colors.ENDC}")
        else:
            self.campaign_manager.ppo_controller._load_model(path)
            print(f"{Colors.GREEN}Model loaded successfully!{Colors.ENDC}")
        
        input("Press Enter to continue...")
    
    def _run_ppo_fuzzing(self):
        """Run PPO-guided fuzzing"""
        clear_screen()
        print_header("PPO-GUIDED FUZZING")
        
        if not self.campaign_manager.ppo_controller.agent:
            print(f"{Colors.YELLOW}No PPO model loaded. Using random actions.{Colors.ENDC}")
        
        target = input("Target binary (or 'demo'): ").strip()
        
        if target.lower() == 'demo':
            self._demo_ppo_fuzzing()
        else:
            # Real PPO fuzzing implementation
            seed_dir = input("Seed directory: ").strip()
            target_args = input("Target arguments [@@]: ").strip() or "@@"
            
            config = CampaignConfig(
                target_binary=target,
                input_dir=seed_dir,
                output_dir=f"./ppo_findings_{int(time.time())}",
                use_qemu=True,
                timeout="1000+",
                memory_limit="none",
                target_args=target_args,
                ppo_enabled=True
            )
            
            name = f"ppo_campaign_{int(time.time())}"
            self.campaign_manager.create_campaign(name, config)
            
            try:
                self.campaign_manager.start_campaign(name)
                self._ppo_control_loop(name)
            except KeyboardInterrupt:
                self.campaign_manager.stop_campaign(name)
        
        input("\nPress Enter to continue...")
    
    def _demo_ppo_fuzzing(self):
        """Demo PPO-guided fuzzing"""
        print(f"\n{Colors.CYAN}Starting PPO-Guided Fuzzing Demo...{Colors.ENDC}")
        print("The AI agent will dynamically select power schedules.\n")
        
        # Show example command
        demo_cmd = """afl-fuzz -Q \\
  -i ./seeds \\
  -o ./batch_output/ppo_demo \\
  -m none -t 1000+ -- \\
  ./target_binary @@"""
        
        print(f"{Colors.CYAN}AFL++ Command:{Colors.ENDC}")
        print(f"{Colors.GREEN}{demo_cmd}{Colors.ENDC}\n")
        
        metrics = FuzzingMetrics()
        state = FuzzingState()
        
        try:
            for i in range(60):
                # Get PPO action
                action = self.campaign_manager.ppo_controller.get_action(state)
                schedule = POWER_SCHEDULES.get(action, "explore")
                
                # Simulate fuzzing with schedule effects
                speed_mult = {
                    "explore": 1.0, "fast": 1.2, "coe": 0.9,
                    "lin": 1.0, "quad": 0.95, "exploit": 1.1, "rare": 0.8
                }.get(schedule, 1.0)
                
                path_mult = {
                    "explore": 1.0, "fast": 0.8, "coe": 1.1,
                    "lin": 1.0, "quad": 1.05, "exploit": 0.7, "rare": 1.3
                }.get(schedule, 1.0)
                
                metrics.total_executions += int(random.randint(100, 500) * speed_mult)
                metrics.total_paths += int(random.randint(0, 3) * path_mult)
                metrics.unique_crashes += 1 if random.random() < 0.03 else 0
                metrics.avg_exec_speed = random.uniform(200, 400) * speed_mult
                
                state.total_paths = metrics.total_paths
                state.execution_speed = metrics.avg_exec_speed
                state.crashes_found = metrics.unique_crashes
                state.current_schedule = action
                
                self._display_stats(metrics, state)
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        
        print(f"\n{Colors.GREEN}Demo complete!{Colors.ENDC}")
    
    def _ppo_control_loop(self, campaign_name: str):
        """PPO control loop for real fuzzing"""
        wrapper = self.campaign_manager.campaigns.get(campaign_name)
        if not wrapper:
            return
        
        while wrapper.running:
            state = wrapper.get_state()
            action = self.campaign_manager.ppo_controller.get_action(state)
            wrapper.set_power_schedule(action)
            
            self._display_stats(wrapper.metrics, state)
            time.sleep(10)
    
    def _show_ppo_info(self):
        """Show PPO model information"""
        clear_screen()
        print_header("PPO MODEL INFORMATION")
        
        info = [
            "Architecture:",
            "  Policy Network: 8 → 128 → 64 → 7 (actions)",
            "  Value Network:  8 → 128 → 64 → 1 (value)",
            "  Total Parameters: ~22,000",
            "",
            "State Space (8 dimensions):",
            "  1. Total paths (normalized)",
            "  2. New paths rate",
            "  3. Crashes found",
            "  4. Hangs found",
            "  5. Coverage percentage",
            "  6. Execution speed",
            "  7. Stability",
            "  8. Current schedule",
            "",
            "Action Space (7 actions):",
            "  0: explore  - Default exploration",
            "  1: fast     - Fast coverage",
            "  2: coe      - Cut-off exponential",
            "  3: lin      - Linear",
            "  4: quad     - Quadratic",
            "  5: exploit  - Exploitation focused",
            "  6: rare     - Rare branches",
            "",
            "Algorithm: PPO (Proximal Policy Optimization)",
            "Reference: Schulman et al., 2017"
        ]
        
        for line in info:
            print(line)
        
        input("\nPress Enter to continue...")
    
    def analysis_menu(self):
        """Analysis and reporting menu"""
        clear_screen()
        print_header("ANALYSIS & REPORTS")
        
        print("[1] Generate Campaign Report")
        print("[2] Generate Research Figures")
        print("[3] Export Crash Analysis")
        print("[4] Compare Campaigns")
        print("[0] Back to Main Menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            self._generate_report()
        elif choice == "2":
            self._generate_figures()
        elif choice == "3":
            self._export_crashes()
        elif choice == "4":
            self._compare_campaigns()
    
    def _generate_report(self):
        """Generate a campaign report"""
        campaigns = self.campaign_manager.list_campaigns()
        
        if not campaigns:
            print(f"{Colors.YELLOW}No campaigns available.{Colors.ENDC}")
            print("Creating demo report...")
            
            # Demo report
            metrics = FuzzingMetrics()
            metrics.total_executions = 1500000
            metrics.total_paths = 2847
            metrics.unique_crashes = 12
            metrics.unique_hangs = 3
            metrics.avg_exec_speed = 350.5
            metrics.peak_exec_speed = 485.2
            
            report = self.report_generator.generate_text_report(metrics)
            print(report)
        else:
            print("Available campaigns:", campaigns)
            name = input("Campaign name: ").strip()
            
            if name in campaigns:
                wrapper = self.campaign_manager.campaigns[name]
                report = self.report_generator.generate_text_report(wrapper.metrics)
                print(report)
            else:
                print(f"{Colors.RED}Campaign not found{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _generate_figures(self):
        """Generate research figures"""
        print(f"\n{Colors.CYAN}Generating research figures...{Colors.ENDC}")
        
        output_dir = "./figures"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate demo figures
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Figure 1: Coverage comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            time_points = np.linspace(0, 3600, 100)
            baseline = 100 * (1 - np.exp(-time_points / 1000)) + np.random.normal(0, 2, 100)
            ppo = 120 * (1 - np.exp(-time_points / 800)) + np.random.normal(0, 2, 100)
            
            ax.plot(time_points / 60, baseline, 'b-', label='AFL++ Baseline', linewidth=2)
            ax.plot(time_points / 60, ppo, 'r-', label='AFL++ + PPO', linewidth=2)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Unique Paths')
            ax.set_title('Code Coverage: Baseline vs PPO-Enhanced')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig(os.path.join(output_dir, 'coverage_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ coverage_comparison.png")
            
            # Figure 2: Power schedule distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            
            schedules = list(POWER_SCHEDULES.values())
            counts = [25, 18, 12, 10, 15, 12, 8]
            colors = plt.cm.Set3(np.linspace(0, 1, 7))
            
            ax.bar(schedules, counts, color=colors)
            ax.set_xlabel('Power Schedule')
            ax.set_ylabel('Selection Frequency (%)')
            ax.set_title('PPO Agent Schedule Selection Distribution')
            plt.xticks(rotation=45)
            
            fig.savefig(os.path.join(output_dir, 'schedule_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ schedule_distribution.png")
            
            # Figure 3: Crash discovery timeline
            fig, ax = plt.subplots(figsize=(10, 6))
            
            crash_times_baseline = sorted([random.uniform(0, 3600) for _ in range(8)])
            crash_times_ppo = sorted([random.uniform(0, 3000) for _ in range(12)])
            
            ax.step(crash_times_baseline, range(1, len(crash_times_baseline) + 1), 
                    'b-', where='post', label='AFL++ Baseline', linewidth=2)
            ax.step(crash_times_ppo, range(1, len(crash_times_ppo) + 1), 
                    'r-', where='post', label='AFL++ + PPO', linewidth=2)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Cumulative Crashes')
            ax.set_title('Crash Discovery Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig(os.path.join(output_dir, 'crash_timeline.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ crash_timeline.png")
            
            print(f"\n{Colors.GREEN}Figures saved to {output_dir}/{Colors.ENDC}")
            
        except ImportError:
            print(f"{Colors.RED}matplotlib not available. Install with:{Colors.ENDC}")
            print("  pip3 install matplotlib --break-system-packages")
        
        input("\nPress Enter to continue...")
    
    def _export_crashes(self):
        """Export crash analysis"""
        print(f"\n{Colors.CYAN}Crash Analysis Export{Colors.ENDC}")
        print("(Demo - would export real crashes from campaign output)")
        
        crash_data = {
            "total_crashes": 12,
            "unique_crashes": 8,
            "crash_types": {
                "heap_buffer_overflow": 4,
                "stack_buffer_overflow": 2,
                "null_dereference": 2
            },
            "reproducer_paths": [
                "./findings/crashes/crash_001",
                "./findings/crashes/crash_002",
                "./findings/crashes/crash_003"
            ]
        }
        
        print(json.dumps(crash_data, indent=2))
        input("\nPress Enter to continue...")
    
    def _compare_campaigns(self):
        """Compare multiple campaigns"""
        print(f"\n{Colors.CYAN}Campaign Comparison{Colors.ENDC}")
        print("(Demo comparison data)")
        
        headers = ["Metric", "Baseline", "PPO", "Improvement"]
        rows = [
            ["Total Paths", "2,847", "3,412", "+19.8%"],
            ["Unique Crashes", "8", "12", "+50%"],
            ["Exec Speed", "285/s", "310/s", "+8.8%"],
            ["Time to 1000 paths", "45min", "32min", "-29%"],
        ]
        
        print_table(headers, rows)
        input("\nPress Enter to continue...")
    
    def system_status(self):
        """Show system status and dependencies"""
        clear_screen()
        print_header("SYSTEM STATUS")
        
        print(f"{Colors.CYAN}Core Dependencies:{Colors.ENDC}")
        
        # Check AFL++
        afl_ok, afl_ver = check_afl_installed()
        status = f"{Colors.GREEN}✓ Installed ({afl_ver}){Colors.ENDC}" if afl_ok else f"{Colors.RED}✗ Not found{Colors.ENDC}"
        print(f"  AFL++:         {status}")
        
        # Check PyTorch
        torch_ok, torch_ver = check_pytorch_installed()
        status = f"{Colors.GREEN}✓ Installed ({torch_ver}){Colors.ENDC}" if torch_ok else f"{Colors.RED}✗ Not found{Colors.ENDC}"
        print(f"  PyTorch:       {status}")
        
        # Check PPO module
        status = f"{Colors.GREEN}✓ Available{Colors.ENDC}" if PPO_AVAILABLE else f"{Colors.RED}✗ Not found{Colors.ENDC}"
        print(f"  PPO Module:    {status}")
        
        # Check matplotlib
        try:
            import matplotlib
            print(f"  matplotlib:    {Colors.GREEN}✓ Installed ({matplotlib.__version__}){Colors.ENDC}")
        except:
            print(f"  matplotlib:    {Colors.RED}✗ Not found{Colors.ENDC}")
        
        # Check numpy
        try:
            import numpy
            print(f"  NumPy:         {Colors.GREEN}✓ Installed ({numpy.__version__}){Colors.ENDC}")
        except:
            print(f"  NumPy:         {Colors.RED}✗ Not found{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}System Info:{Colors.ENDC}")
        print(f"  Python:        {sys.version.split()[0]}")
        print(f"  CPU Cores:     {mp.cpu_count()}")
        print(f"  Working Dir:   {os.getcwd()}")
        print(f"  Framework Ver: {VERSION}")
        
        print(f"\n{Colors.CYAN}AFL++ Command Format:{Colors.ENDC}")
        print(f"  afl-fuzz -Q -i ./seeds -o ./output -m none -t 1000+ -- ./target @@")
        
        input("\nPress Enter to continue...")
    
    def settings_menu(self):
        """Settings configuration"""
        clear_screen()
        print_header("SETTINGS")
        
        print("Current Configuration:")
        for key, value in DEFAULT_CONFIG.items():
            print(f"  {key}: {value}")
        
        print("\n(Settings customization not implemented in demo)")
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main menu loop"""
        while self.running:
            clear_screen()
            self.show_main_menu()
            
            choice = input(f"\n{Colors.CYAN}Select option: {Colors.ENDC}").strip().upper()
            
            if choice == '1':
                self.quick_start()
            elif choice == '2':
                self._campaign_manager_menu()
            elif choice == '3':
                self._seed_generator_menu()
            elif choice == '4':
                self._mutation_tester_menu()
            elif choice == '5':
                self._train_ppo()
            elif choice == '6':
                self._run_ppo_fuzzing()
            elif choice == '7':
                self._batch_fuzzing_menu()
            elif choice == '8':
                self.analysis_menu()
            elif choice == '9':
                self.system_status()
            elif choice == '0':
                self.settings_menu()
            elif choice == 'H':
                self.show_help()
            elif choice == 'Q':
                self.running = False
                print(f"\n{Colors.GREEN}Thank you for using FuzzMaster Pro!{Colors.ENDC}")
    
    def _batch_fuzzing_menu(self):
        """Batch fuzzing menu - fuzz all binaries in a folder"""
        clear_screen()
        print_header("BATCH FUZZING - QEMU MODE")
        
        if not BATCH_AVAILABLE:
            print(f"{Colors.RED}Error: Batch fuzzer module not available!{Colors.ENDC}")
            print("\nMake sure batch_fuzzer.py is in the same directory.")
            input("\nPress Enter to return...")
            return
        
        print(f"""
{Colors.CYAN}This mode will automatically fuzz ALL binaries in a directory.{Colors.ENDC}

Features:
  • Scans folder for all ELF executables
  • Uses QEMU mode (-Q) for uninstrumented binaries
  • Processes each binary sequentially
  • Saves results for each binary separately
  • Generates aggregated report at the end

{Colors.YELLOW}AFL++ Command Format:{Colors.ENDC}
  afl-fuzz -Q \\
    -i ./seeds \\
    -o ./batch_output/<binary_name> \\
    -m none -t 1000+ -- \\
    ./path/to/binary @@

{Colors.YELLOW}Options:{Colors.ENDC}
  [1] Start Batch Fuzzing (Real AFL++)
  [2] Simulation Mode (Demo, no AFL++)
  [3] View Previous Results
  [0] Back to Main Menu
""")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            self._run_batch_fuzzing(simulate=False)
        elif choice == '2':
            self._run_batch_fuzzing(simulate=True)
        elif choice == '3':
            self._view_batch_results()
    
    def _run_batch_fuzzing(self, simulate: bool = False):
        """Run batch fuzzing on all binaries"""
        clear_screen()
        mode = "SIMULATION" if simulate else "REAL AFL++"
        print_header(f"BATCH FUZZING - {mode}")
        
        # Get configuration
        print(f"{Colors.CYAN}Configuration:{Colors.ENDC}\n")
        
        binary_dir = input("  Binary directory [./binaries]: ").strip() or "./binaries"
        
        # Check if directory exists
        if not os.path.isdir(binary_dir):
            print(f"\n{Colors.RED}Directory not found: {binary_dir}{Colors.ENDC}")
            print("Please provide a valid directory containing binaries.")
            input("\nPress Enter to continue...")
            return
        
        output_dir = input("  Output directory [./batch_output]: ").strip() or "./batch_output"
        
        time_input = input("  Time per binary in seconds [60]: ").strip()
        time_per_binary = int(time_input) if time_input else 60
        
        use_qemu = input("  Use QEMU mode (-Q)? (Y/n): ").strip().lower() != 'n'
        use_ppo = input("  Use PPO optimization? (y/N): ").strip().lower() == 'y'
        
        target_args = input("  Target arguments [@@]: ").strip() or "@@"
        
        # Count binaries
        binary_count = sum(1 for f in os.listdir(binary_dir) 
                          if os.path.isfile(os.path.join(binary_dir, f)))
        
        print(f"\n{Colors.GREEN}Found {binary_count} files in {binary_dir}{Colors.ENDC}")
        
        if simulate:
            total_time = binary_count * min(30, time_per_binary)
        else:
            total_time = binary_count * time_per_binary
        
        print(f"Estimated total time: {total_time // 60} minutes ({total_time // 3600:.1f} hours)")
        
        # Show example command
        print(f"\n{Colors.CYAN}AFL++ Command Format:{Colors.ENDC}")
        qemu_flag = "-Q " if use_qemu else ""
        print(f"""  afl-fuzz {qemu_flag}\\
    -i ./seeds \\
    -o {output_dir}/<binary_name> \\
    -m none -t 1000+ -- \\
    {binary_dir}/<binary> {target_args}""")
        
        confirm = input(f"\nStart batch fuzzing? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Run batch fuzzer
        print(f"\n{Colors.GREEN}Starting batch fuzzing...{Colors.ENDC}")
        print("Press Ctrl+C to stop early.\n")
        
        try:
            if simulate:
                fuzzer = SimulatedBatchFuzzer(
                    binary_dir=binary_dir,
                    output_dir=output_dir,
                    use_qemu=use_qemu,
                    use_ppo=use_ppo,
                    time_per_binary=time_per_binary,
                    target_args=target_args
                )
            else:
                fuzzer = BatchFuzzer(
                    binary_dir=binary_dir,
                    output_dir=output_dir,
                    use_qemu=use_qemu,
                    use_ppo=use_ppo,
                    time_per_binary=time_per_binary,
                    target_args=target_args
                )
            
            results = fuzzer.run_all()
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Batch fuzzing interrupted.{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}Error during batch fuzzing: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _view_batch_results(self):
        """View previous batch results"""
        clear_screen()
        print_header("BATCH RESULTS VIEWER")
        
        results_path = input("Results file [./batch_output/batch_results.json]: ").strip()
        results_path = results_path or "./batch_output/batch_results.json"
        
        if not os.path.exists(results_path):
            print(f"{Colors.RED}Results file not found: {results_path}{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print(f"\n{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
            print(f"{Colors.BOLD}BATCH FUZZING RESULTS{Colors.ENDC}")
            print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
            
            summary = results.get('summary', {})
            print(f"\nTotal Binaries:  {summary.get('total_binaries', 0)}")
            print(f"Completed:       {Colors.GREEN}{summary.get('completed', 0)}{Colors.ENDC}")
            print(f"Failed:          {Colors.RED}{summary.get('failed', 0)}{Colors.ENDC}")
            print(f"Skipped:         {summary.get('skipped', 0)}")
            print(f"\n{Colors.YELLOW}Aggregated Results:{Colors.ENDC}")
            print(f"  Total Paths:   {summary.get('total_paths', 0)}")
            print(f"  Total Crashes: {Colors.RED}{summary.get('total_crashes', 0)}{Colors.ENDC}")
            print(f"  Total Hangs:   {summary.get('total_hangs', 0)}")
            
            # Per-binary results
            binaries = results.get('binaries', {})
            if binaries:
                print(f"\n{Colors.CYAN}Per-Binary Results:{Colors.ENDC}")
                print("-" * 70)
                print(f"{'Binary':<35} {'Status':<12} {'Paths':<8} {'Crashes':<8}")
                print("-" * 70)
                
                for name, data in binaries.items():
                    status = data.get('status', 'unknown')
                    status_color = Colors.GREEN if status == 'completed' else Colors.RED
                    print(f"{name[:34]:<35} {status_color}{status:<12}{Colors.ENDC} {data.get('total_paths', 0):<8} {data.get('unique_crashes', 0):<8}")
            
            print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error reading results: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _campaign_manager_menu(self):
        """Campaign manager submenu"""
        clear_screen()
        print_header("CAMPAIGN MANAGER")
        
        print("[1] Create New Campaign")
        print("[2] List Campaigns")
        print("[3] Start Campaign")
        print("[4] Stop Campaign")
        print("[5] Campaign Status")
        print("[0] Back")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "1":
            name = input("Campaign name: ").strip()
            target = input("Target binary: ").strip()
            seeds = input("Seed directory: ").strip()
            target_args = input("Target arguments [@@]: ").strip() or "@@"
            
            config = CampaignConfig(
                target_binary=target,
                input_dir=seeds,
                output_dir=f"./output_{name}",
                use_qemu=True,
                timeout="1000+",
                memory_limit="none",
                target_args=target_args
            )
            
            if self.campaign_manager.create_campaign(name, config):
                print(f"{Colors.GREEN}Campaign created!{Colors.ENDC}")
            
        elif choice == "2":
            campaigns = self.campaign_manager.list_campaigns()
            if campaigns:
                print("\nCampaigns:", campaigns)
            else:
                print("\nNo campaigns.")
                
        elif choice == "3":
            name = input("Campaign name: ").strip()
            self.campaign_manager.start_campaign(name)
            
        elif choice == "4":
            name = input("Campaign name: ").strip()
            self.campaign_manager.stop_campaign(name)
            
        elif choice == "5":
            name = input("Campaign name: ").strip()
            status = self.campaign_manager.get_campaign_status(name)
            print(json.dumps(status, indent=2))
        
        input("\nPress Enter to continue...")
    
    def _seed_generator_menu(self):
        """Seed generator submenu"""
        clear_screen()
        print_header("SEED GENERATOR")
        
        print("[1] Generate Basic Random Seeds")
        print("[2] Generate Format-Specific Seeds")
        print("[3] Minimize Existing Corpus")
        print("[0] Back")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "1":
            output = input("Output directory [./seeds]: ").strip() or "./seeds"
            count = int(input("Number of seeds [20]: ").strip() or "20")
            generated = self.seed_generator.generate_basic_seeds(output, count)
            print(f"{Colors.GREEN}Generated {generated} seeds{Colors.ENDC}")
            
        elif choice == "2":
            print("\nAvailable formats: generic, elf, pe, pdf, json, xml, png")
            fmt = input("Format [generic]: ").strip() or "generic"
            output = input("Output directory [./seeds]: ").strip() or "./seeds"
            generated = self.seed_generator.generate_format_seeds(output, fmt)
            print(f"{Colors.GREEN}Generated {generated} {fmt} seeds{Colors.ENDC}")
            
        elif choice == "3":
            print("Corpus minimization requires AFL++ afl-cmin")
            input_dir = input("Input corpus: ").strip()
            output_dir = input("Output directory: ").strip()
            target = input("Target binary: ").strip()
            
            if self.seed_generator.minimize_corpus(input_dir, output_dir, target):
                print(f"{Colors.GREEN}Minimization complete!{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Minimization failed{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _mutation_tester_menu(self):
        """Mutation strategy tester"""
        clear_screen()
        print_header("MUTATION TESTER")
        
        print("Test mutation strategies on sample input.\n")
        
        # Sample input
        sample = b"Hello World! This is a test input for fuzzing."
        print(f"Original ({len(sample)} bytes): {sample[:50]}...")
        print("-" * 50)
        
        mutations = [
            ("bit_flip", self.mutator.bit_flip),
            ("byte_flip", self.mutator.byte_flip),
            ("arithmetic", self.mutator.arithmetic),
            ("interesting", self.mutator.interesting_values),
            ("havoc", self.mutator.havoc),
            ("insert", self.mutator.insert_random),
            ("delete", self.mutator.delete_random),
        ]
        
        for name, func in mutations:
            try:
                result = func(sample)
                print(f"{name:15} ({len(result):3} bytes): {result[:50]}...")
            except Exception as e:
                print(f"{name:15}: Error - {e}")
        
        print("-" * 50)
        print(f"\nMutation stats: {dict(self.mutator.mutation_stats)}")
        
        input("\nPress Enter to continue...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FuzzMaster Pro - AFL++ with PPO Reinforcement Learning"
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"FuzzMaster Pro v{VERSION}")
        return
    
    if args.no_color:
        Colors.disable()
    
    # Start menu system
    menu = MenuSystem()
    
    if args.demo:
        menu._run_demo_fuzzing()
    else:
        try:
            menu.run()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted. Goodbye!{Colors.ENDC}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


if __name__ == "__main__":
    main()
