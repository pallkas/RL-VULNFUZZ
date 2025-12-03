#!/usr/bin/env python3
"""
Batch Fuzzer Module for RL-VULNFUZZ Framework
Fuzzes all binaries in a directory using AFL++ with QEMU mode

Command Format:
    afl-fuzz -Q \
        -i ./seeds \
        -o ./batch_output/<binary_name> \
        -m none -t 1000+ -- \
        ./path/to/binary [args] @@
"""

import os
import sys
import time
import json
import signal
import random
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("BatchFuzzer")

# SPEC CPU2006 binary argument mappings
SPEC2006_BINARY_ARGS = {
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
}


def get_binary_args(binary_path: str, custom_args: str = None) -> str:
    """
    Get the appropriate arguments for a binary.
    """
    # If custom args provided and not default, use them
    if custom_args and custom_args.strip() and custom_args.strip() != "@@":
        return custom_args
    
    binary_name = os.path.basename(binary_path)
    
    # Remove common suffixes
    for suffix in ['.exe', '_base', '_peak', '_O2', '_O3', '_r', '_s']:
        if binary_name.endswith(suffix):
            binary_name = binary_name[:-len(suffix)]
    
    # Handle names like "464.h264ref" or "libquantum_base.amd64-m64-gcc42-nn"
    # Extract the core binary name
    if '.' in binary_name:
        parts = binary_name.split('.')
        if parts[0].isdigit():
            binary_name = parts[1] if len(parts) > 1 else parts[0]
        else:
            binary_name = parts[0]
    
    # Handle underscores in names like "libquantum_base"
    if '_' in binary_name:
        binary_name = binary_name.split('_')[0]
    
    # Check for known mapping
    if binary_name in SPEC2006_BINARY_ARGS:
        return SPEC2006_BINARY_ARGS[binary_name]
    
    # Partial match
    for known_binary, args in SPEC2006_BINARY_ARGS.items():
        if known_binary in binary_name or binary_name in known_binary:
            return args
    
    return "@@"


@dataclass
class BatchConfig:
    """Configuration for batch fuzzing"""
    binary_dir: str
    output_dir: str
    seed_dir: str = "./seeds"
    use_qemu: bool = True
    use_ppo: bool = False
    time_per_binary: int = 60
    timeout: str = "1000+"      # With + suffix
    memory_limit: str = "none"  # No memory limit
    target_args: str = "@@"


@dataclass
class BinaryResult:
    """Result from fuzzing a single binary"""
    binary_name: str
    binary_path: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_paths: int = 0
    unique_crashes: int = 0
    unique_hangs: int = 0
    exec_speed: float = 0.0
    error_message: str = ""
    command_used: str = ""


class BatchFuzzer:
    """
    Batch fuzzer for multiple binaries using AFL++ with QEMU mode.
    
    Command format:
        afl-fuzz -Q -i ./seeds -o ./output -m none -t 1000+ -- ./binary @@
    """
    
    def __init__(self, binary_dir: str, output_dir: str, seed_dir: str = "./seeds",
                 use_qemu: bool = True, use_ppo: bool = False, 
                 time_per_binary: int = 60, target_args: str = "@@"):
        self.binary_dir = binary_dir
        self.output_dir = output_dir
        self.seed_dir = seed_dir
        self.use_qemu = use_qemu
        self.use_ppo = use_ppo
        self.time_per_binary = time_per_binary
        self.target_args = target_args
        
        # Fixed settings for correct AFL++ syntax
        self.timeout = "1000+"
        self.memory_limit = "none"
        
        self.results: Dict[str, BinaryResult] = {}
        self.current_process: Optional[subprocess.Popen] = None
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(seed_dir, exist_ok=True)
        
        # Generate seeds if directory is empty
        if not os.listdir(seed_dir):
            self._generate_seeds()
    
    def _generate_seeds(self):
        """Generate basic seed files"""
        logger.info(f"Generating seeds in {self.seed_dir}")
        for i in range(10):
            seed_path = os.path.join(self.seed_dir, f"seed_{i:04d}")
            with open(seed_path, 'wb') as f:
                f.write(os.urandom(random.randint(16, 256)))
    
    def discover_binaries(self) -> List[str]:
        """Find all executable binaries in the binary directory"""
        binaries = []
        
        for item in os.listdir(self.binary_dir):
            item_path = os.path.join(self.binary_dir, item)
            
            if os.path.isfile(item_path):
                # Check if executable
                if os.access(item_path, os.X_OK):
                    binaries.append(item_path)
                else:
                    # Try to check if it's an ELF file
                    try:
                        with open(item_path, 'rb') as f:
                            magic = f.read(4)
                            if magic == b'\x7fELF':
                                binaries.append(item_path)
                    except:
                        pass
        
        logger.info(f"Discovered {len(binaries)} binaries in {self.binary_dir}")
        return sorted(binaries)
    
    def build_command(self, binary_path: str) -> List[str]:
        """
        Build AFL++ command for a binary.
        
        Format: afl-fuzz -Q -i ./seeds -o ./output/<name> -m none -t 1000+ -- ./binary [args] @@
        """
        binary_name = os.path.basename(binary_path)
        # Clean up binary name for output directory
        safe_name = binary_name.replace('.', '_').replace('-', '_')
        binary_output = os.path.join(self.output_dir, safe_name)
        
        cmd = ["afl-fuzz"]
        
        # QEMU mode FIRST (important!)
        if self.use_qemu:
            cmd.append("-Q")
        
        # Input directory
        cmd.extend(["-i", self.seed_dir])
        
        # Output directory
        cmd.extend(["-o", binary_output])
        
        # Memory limit - must be "none"
        cmd.extend(["-m", self.memory_limit])
        
        # Timeout with + suffix
        cmd.extend(["-t", self.timeout])
        
        # Separator
        cmd.append("--")
        
        # Target binary
        cmd.append(binary_path)
        
        # Binary-specific arguments
        args = get_binary_args(binary_path, self.target_args)
        if args:
            cmd.extend(args.split())
        
        return cmd
    
    def get_command_string(self, binary_path: str) -> str:
        """Get formatted command string for display"""
        cmd = self.build_command(binary_path)
        return " ".join(cmd)
    
    def get_command_display(self, binary_path: str) -> str:
        """Get nicely formatted command for display"""
        binary_name = os.path.basename(binary_path)
        safe_name = binary_name.replace('.', '_').replace('-', '_')
        binary_output = os.path.join(self.output_dir, safe_name)
        
        args = get_binary_args(binary_path, self.target_args)
        
        lines = [
            "afl-fuzz -Q \\",
            f"  -i {self.seed_dir} \\",
            f"  -o {binary_output} \\",
            f"  -m {self.memory_limit} -t {self.timeout} -- \\",
            f"  {binary_path} {args}"
        ]
        return "\n".join(lines)
    
    def fuzz_binary(self, binary_path: str) -> BinaryResult:
        """Fuzz a single binary"""
        binary_name = os.path.basename(binary_path)
        result = BinaryResult(binary_name=binary_name, binary_path=binary_path)
        result.start_time = datetime.now()
        
        cmd = self.build_command(binary_path)
        result.command_used = " ".join(cmd)
        
        logger.info(f"Fuzzing: {binary_name}")
        logger.info(f"Command: {result.command_used}")
        
        # Print formatted command
        print(f"\n{'='*70}")
        print(f"Target: {binary_name}")
        print(f"{'='*70}")
        print(self.get_command_display(binary_path))
        print(f"{'='*70}\n")
        
        try:
            # Create output directory
            safe_name = binary_name.replace('.', '_').replace('-', '_')
            binary_output = os.path.join(self.output_dir, safe_name)
            os.makedirs(binary_output, exist_ok=True)
            
            # Start AFL++
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Run for specified time
            try:
                self.current_process.wait(timeout=self.time_per_binary)
            except subprocess.TimeoutExpired:
                # Expected - we stop after time limit
                pass
            
            # Stop the process
            self._stop_current_process()
            
            # Parse results
            result = self._parse_results(binary_path, result)
            result.status = "completed"
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            logger.error(f"Error fuzzing {binary_name}: {e}")
            self._stop_current_process()
        
        result.end_time = datetime.now()
        self.results[binary_name] = result
        
        return result
    
    def _stop_current_process(self):
        """Stop the current fuzzing process"""
        if self.current_process:
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                except:
                    pass
            self.current_process = None
    
    def _parse_results(self, binary_path: str, result: BinaryResult) -> BinaryResult:
        """Parse fuzzing results from AFL++ output directory"""
        binary_name = os.path.basename(binary_path)
        safe_name = binary_name.replace('.', '_').replace('-', '_')
        binary_output = os.path.join(self.output_dir, safe_name)
        
        # Try to find fuzzer_stats
        stats_paths = [
            os.path.join(binary_output, "default", "fuzzer_stats"),
            os.path.join(binary_output, "fuzzer_stats"),
        ]
        
        for stats_path in stats_paths:
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        for line in f:
                            if ':' in line:
                                key, value = line.strip().split(':', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                if key == 'corpus_count':
                                    result.total_paths = int(value)
                                elif key == 'saved_crashes':
                                    result.unique_crashes = int(value)
                                elif key == 'saved_hangs':
                                    result.unique_hangs = int(value)
                                elif key == 'execs_per_sec':
                                    result.exec_speed = float(value)
                    break
                except Exception as e:
                    logger.debug(f"Error parsing stats: {e}")
        
        # Count crashes and hangs from directories
        crashes_dir = os.path.join(binary_output, "default", "crashes")
        if os.path.isdir(crashes_dir):
            result.unique_crashes = max(result.unique_crashes, 
                                        len([f for f in os.listdir(crashes_dir) if f != "README.txt"]))
        
        hangs_dir = os.path.join(binary_output, "default", "hangs")
        if os.path.isdir(hangs_dir):
            result.unique_hangs = max(result.unique_hangs,
                                      len([f for f in os.listdir(hangs_dir) if f != "README.txt"]))
        
        return result
    
    def run_all(self) -> Dict[str, BinaryResult]:
        """Fuzz all binaries in the directory"""
        binaries = self.discover_binaries()
        
        if not binaries:
            logger.warning(f"No binaries found in {self.binary_dir}")
            return {}
        
        print(f"\n{'='*70}")
        print(f"BATCH FUZZING - {len(binaries)} binaries")
        print(f"Time per binary: {self.time_per_binary} seconds")
        print(f"Total estimated time: {len(binaries) * self.time_per_binary // 60} minutes")
        print(f"{'='*70}\n")
        
        for i, binary_path in enumerate(binaries, 1):
            binary_name = os.path.basename(binary_path)
            print(f"\n[{i}/{len(binaries)}] Processing: {binary_name}")
            
            try:
                result = self.fuzz_binary(binary_path)
                
                # Print result summary
                status_color = "\033[92m" if result.status == "completed" else "\033[91m"
                print(f"{status_color}Status: {result.status}\033[0m")
                print(f"Paths: {result.total_paths} | Crashes: {result.unique_crashes} | Hangs: {result.unique_hangs}")
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self._stop_current_process()
                break
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON file"""
        results_path = os.path.join(self.output_dir, "batch_results.json")
        
        # Calculate summary
        total_paths = sum(r.total_paths for r in self.results.values())
        total_crashes = sum(r.unique_crashes for r in self.results.values())
        total_hangs = sum(r.unique_hangs for r in self.results.values())
        completed = sum(1 for r in self.results.values() if r.status == "completed")
        failed = sum(1 for r in self.results.values() if r.status == "error")
        
        data = {
            "generated": datetime.now().isoformat(),
            "config": {
                "binary_dir": self.binary_dir,
                "output_dir": self.output_dir,
                "use_qemu": self.use_qemu,
                "time_per_binary": self.time_per_binary,
                "timeout": self.timeout,
                "memory_limit": self.memory_limit,
            },
            "summary": {
                "total_binaries": len(self.results),
                "completed": completed,
                "failed": failed,
                "skipped": len(self.results) - completed - failed,
                "total_paths": total_paths,
                "total_crashes": total_crashes,
                "total_hangs": total_hangs,
            },
            "binaries": {
                name: {
                    "binary_path": r.binary_path,
                    "status": r.status,
                    "total_paths": r.total_paths,
                    "unique_crashes": r.unique_crashes,
                    "unique_hangs": r.unique_hangs,
                    "exec_speed": r.exec_speed,
                    "error_message": r.error_message,
                    "command_used": r.command_used,
                }
                for name, r in self.results.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("BATCH FUZZING COMPLETE")
        print(f"{'='*70}")
        print(f"Total binaries: {len(self.results)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Total paths: {total_paths}")
        print(f"Total crashes: {total_crashes}")
        print(f"Total hangs: {total_hangs}")
        print(f"Results: {results_path}")
        print(f"{'='*70}\n")


class SimulatedBatchFuzzer(BatchFuzzer):
    """Simulated batch fuzzer for demo/testing without actual AFL++"""
    
    def fuzz_binary(self, binary_path: str) -> BinaryResult:
        """Simulate fuzzing a binary"""
        binary_name = os.path.basename(binary_path)
        result = BinaryResult(binary_name=binary_name, binary_path=binary_path)
        result.start_time = datetime.now()
        result.command_used = self.get_command_string(binary_path)
        
        print(f"\n{'='*70}")
        print(f"[SIMULATION] Target: {binary_name}")
        print(f"{'='*70}")
        print(self.get_command_display(binary_path))
        print(f"{'='*70}")
        
        # Simulate fuzzing
        sim_time = min(30, self.time_per_binary)
        for i in range(sim_time):
            progress = (i + 1) / sim_time * 100
            paths = random.randint(0, 3) * (i + 1)
            crashes = 1 if random.random() < 0.05 else 0
            
            print(f"\r[{'█' * int(progress/5)}{'░' * (20 - int(progress/5))}] {progress:.0f}% | "
                  f"Paths: {paths} | Crashes: {result.unique_crashes}", end="")
            
            result.total_paths = max(result.total_paths, paths)
            result.unique_crashes += crashes
            time.sleep(1)
        
        print()  # New line after progress bar
        
        result.status = "completed"
        result.end_time = datetime.now()
        result.exec_speed = random.uniform(100, 500)
        
        self.results[binary_name] = result
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch AFL++ Fuzzer")
    parser.add_argument("binary_dir", help="Directory containing binaries")
    parser.add_argument("-o", "--output", default="./batch_output", help="Output directory")
    parser.add_argument("-i", "--input", default="./seeds", help="Seed directory")
    parser.add_argument("-t", "--time", type=int, default=60, help="Time per binary (seconds)")
    parser.add_argument("--no-qemu", action="store_true", help="Disable QEMU mode")
    parser.add_argument("--simulate", action="store_true", help="Simulation mode (no AFL++)")
    
    args = parser.parse_args()
    
    if args.simulate:
        fuzzer = SimulatedBatchFuzzer(
            binary_dir=args.binary_dir,
            output_dir=args.output,
            seed_dir=args.input,
            use_qemu=not args.no_qemu,
            time_per_binary=args.time
        )
    else:
        fuzzer = BatchFuzzer(
            binary_dir=args.binary_dir,
            output_dir=args.output,
            seed_dir=args.input,
            use_qemu=not args.no_qemu,
            time_per_binary=args.time
        )
    
    try:
        fuzzer.run_all()
    except KeyboardInterrupt:
        print("\nInterrupted.")
