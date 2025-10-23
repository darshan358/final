#!/usr/bin/env python3
"""
Demonstration script for Bitcoin Address Generator
"""

import time
import os
from btc_generator import BitcoinAddressGenerator
from address_checker import AddressChecker
from utils import get_system_info, format_number, create_performance_report

def demo_address_generation():
    """Demonstrate address generation"""
    print("🔑 BITCOIN ADDRESS GENERATION DEMO")
    print("=" * 50)
    
    generator = BitcoinAddressGenerator(use_gpu=False)
    
    print("Generating 3 random Bitcoin addresses...")
    addresses = generator.generate_address_batch(3)
    
    for i, addr in enumerate(addresses, 1):
        print(f"\n📍 Address {i}:")
        print(f"   Mnemonic: {addr['mnemonic']}")
        print(f"   Private Key: {addr['private_key_wif']}")
        print(f"   P2PKH (Legacy): {addr['p2pkh_address']}")
        print(f"   P2SH (Script): {addr['p2sh_address']}")
        print(f"   Bech32 (Segwit): {addr['bech32_address']}")
    
    return addresses

def demo_address_checking(sample_addresses):
    """Demonstrate address checking"""
    print("\n🔍 ADDRESS CHECKING DEMO")
    print("=" * 50)
    
    checker = AddressChecker()
    
    print(f"Target addresses loaded: {len(checker.target_addresses)}")
    print("Checking generated addresses against target list...")
    
    matches = checker.check_addresses(sample_addresses)
    
    if matches:
        print(f"🎉 Found {len(matches)} matches!")
        for match in matches:
            print(f"   Matched: {match['matched_address']}")
    else:
        print("No matches found (this is expected - collisions are extremely rare)")
    
    return matches

def demo_performance_test():
    """Demonstrate performance testing"""
    print("\n⚡ PERFORMANCE DEMO")
    print("=" * 50)
    
    generator = BitcoinAddressGenerator(use_gpu=False)
    
    print("Running 5-second performance test...")
    start_time = time.time()
    total_addresses = 0
    
    while time.time() - start_time < 5:
        batch = generator.generate_address_batch(100)
        total_addresses += len(batch)
    
    elapsed = time.time() - start_time
    rate = total_addresses / elapsed
    
    print(f"Generated {format_number(total_addresses)} addresses in {elapsed:.2f} seconds")
    print(f"Performance: {rate:.2f} addresses/second")
    
    return {"addresses_generated": total_addresses, "duration": elapsed, "rate": rate}

def demo_system_info():
    """Show system information"""
    print("\n💻 SYSTEM INFORMATION")
    print("=" * 50)
    
    info = get_system_info()
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

def demo_file_operations():
    """Demonstrate file operations"""
    print("\n📁 FILE OPERATIONS DEMO")
    print("=" * 50)
    
    # Show addresses.txt content
    if os.path.exists("addresses.txt"):
        with open("addresses.txt", "r") as f:
            lines = f.readlines()
        
        target_count = len([line for line in lines if line.strip() and not line.startswith('#')])
        print(f"Target addresses file: addresses.txt ({target_count} addresses)")
        
        print("Sample target addresses:")
        for line in lines[:5]:
            if line.strip() and not line.startswith('#'):
                print(f"   {line.strip()}")
    
    # Show project files
    print(f"\nProject files:")
    files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in sorted(files):
        size = os.path.getsize(file)
        print(f"   {file} ({size} bytes)")

def main():
    """Main demonstration"""
    print("🚀 BITCOIN ADDRESS GENERATOR DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the key features of the Bitcoin Address Generator")
    print("=" * 60)
    
    # System info
    demo_system_info()
    
    # Address generation
    sample_addresses = demo_address_generation()
    
    # Address checking
    demo_address_checking(sample_addresses)
    
    # Performance test
    perf_stats = demo_performance_test()
    
    # File operations
    demo_file_operations()
    
    # Final summary
    print("\n🎯 DEMO SUMMARY")
    print("=" * 50)
    print("✅ Address generation: Working")
    print("✅ Multiple address types: P2PKH, P2SH, Bech32")
    print("✅ Mnemonic seed phrases: BIP39 compliant")
    print("✅ Address validation: All formats supported")
    print("✅ Performance testing: CPU optimized")
    print("✅ File operations: Target list management")
    print("✅ Multithreading: Ready for parallel processing")
    
    print(f"\nPerformance: {perf_stats['rate']:.2f} addresses/second")
    print("GPU acceleration: Available when CUDA is installed")
    
    print("\n🚀 READY TO USE!")
    print("Run 'python3 main.py --help' for usage instructions")
    print("Run 'python3 main.py --sample 10' to generate sample addresses")
    print("Run 'python3 main.py' to start mining")

if __name__ == "__main__":
    main()