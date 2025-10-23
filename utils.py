#!/usr/bin/env python3
"""
Utility functions for Bitcoin Address Generator
"""

import json
import time
import os
from typing import Dict, List
import hashlib

def save_config(config: Dict, filename: str = "config.json"):
    """Save configuration to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filename}")
    except Exception as e:
        print(f"Error saving config: {e}")

def load_config(filename: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    default_config = {
        "threads": 8,
        "batch_size": 1000,
        "use_gpu": True,
        "save_interval": 60,
        "stats_interval": 10
    }
    
    if not os.path.exists(filename):
        save_config(default_config, filename)
        return default_config
    
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        # Merge with defaults for missing keys
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"

def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_hash_rate(addresses: int, time_seconds: float) -> float:
    """Calculate hash rate in addresses per second"""
    if time_seconds <= 0:
        return 0.0
    return addresses / time_seconds

def estimate_time_to_collision(total_addresses: int, rate_per_second: float) -> str:
    """Estimate time to find collision (theoretical)"""
    # This is purely theoretical - actual collision probability is astronomically low
    bitcoin_address_space = 2**160  # Approximate address space
    
    if rate_per_second <= 0:
        return "Unknown"
    
    # Using birthday paradox approximation
    import math
    addresses_needed = math.sqrt(math.pi * bitcoin_address_space / 2)
    seconds_needed = addresses_needed / rate_per_second
    
    if seconds_needed > 1e15:  # More than ~31 million years
        return "Longer than age of universe"
    
    return format_time(seconds_needed)

def validate_address_format(address: str) -> bool:
    """Validate Bitcoin address format"""
    if not address:
        return False
    
    # P2PKH (starts with 1)
    if address.startswith('1'):
        return len(address) >= 26 and len(address) <= 35
    
    # P2SH (starts with 3)
    elif address.startswith('3'):
        return len(address) >= 26 and len(address) <= 35
    
    # Bech32 (starts with bc1)
    elif address.startswith('bc1'):
        return len(address) >= 14 and len(address) <= 74
    
    return False

def clean_address_list(addresses: List[str]) -> List[str]:
    """Clean and validate address list"""
    cleaned = []
    for addr in addresses:
        addr = addr.strip()
        if addr and not addr.startswith('#') and validate_address_format(addr):
            cleaned.append(addr)
    return list(set(cleaned))  # Remove duplicates

def get_system_info() -> Dict:
    """Get system information"""
    import platform
    import os
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version()
    }
    
    # Try to get memory info
    try:
        import psutil
        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        info["memory_gb"] = "Unknown (psutil not installed)"
    
    # Check for NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
        else:
            info["gpu"] = "None detected"
    except:
        info["gpu"] = "None detected"
    
    return info

def create_performance_report(stats: Dict, duration: float) -> str:
    """Create performance report"""
    report = []
    report.append("=" * 60)
    report.append("PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Duration: {format_time(duration)}")
    report.append(f"Addresses Generated: {format_number(stats.get('addresses_generated', 0))}")
    report.append(f"Addresses Checked: {format_number(stats.get('addresses_checked', 0))}")
    report.append(f"Matches Found: {stats.get('matches_found', 0)}")
    
    if duration > 0:
        gen_rate = stats.get('addresses_generated', 0) / duration
        check_rate = stats.get('addresses_checked', 0) / duration
        report.append(f"Generation Rate: {gen_rate:.2f} addr/sec")
        report.append(f"Check Rate: {check_rate:.2f} addr/sec")
    
    report.append("=" * 60)
    return "\n".join(report)

def log_event(message: str, level: str = "INFO", filename: str = "mining.log"):
    """Log event to file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"
    
    try:
        with open(filename, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error writing to log: {e}")

def backup_found_addresses(source: str = "found_addresses.txt", backup_dir: str = "backups"):
    """Create backup of found addresses"""
    if not os.path.exists(source):
        return
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"found_addresses_{timestamp}.txt"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        import shutil
        shutil.copy2(source, backup_path)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")

def main():
    """Utility script main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bitcoin Address Generator Utilities")
    parser.add_argument("--system-info", action="store_true", help="Show system information")
    parser.add_argument("--clean-addresses", help="Clean address file")
    parser.add_argument("--backup", action="store_true", help="Backup found addresses")
    parser.add_argument("--config", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.system_info:
        info = get_system_info()
        print("System Information:")
        print("-" * 30)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    if args.clean_addresses:
        try:
            with open(args.clean_addresses, 'r') as f:
                addresses = f.readlines()
            
            cleaned = clean_address_list(addresses)
            
            with open(args.clean_addresses, 'w') as f:
                f.write("# Cleaned Bitcoin addresses\n")
                for addr in sorted(cleaned):
                    f.write(f"{addr}\n")
            
            print(f"Cleaned {len(cleaned)} addresses in {args.clean_addresses}")
        except Exception as e:
            print(f"Error cleaning addresses: {e}")
    
    if args.backup:
        backup_found_addresses()
    
    if args.config:
        config = load_config()
        print("Current Configuration:")
        print("-" * 30)
        for key, value in config.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()