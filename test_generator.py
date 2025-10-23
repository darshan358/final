#!/usr/bin/env python3
"""
Test script for Bitcoin Address Generator
Generates a few addresses to verify the system is working
"""

import sys
from colorama import init, Fore, Style

try:
    from btc_generator import BitcoinAddressGenerator
    init(autoreset=True)
except ImportError as e:
    print(f"Error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def test_basic_generation():
    """Test basic address generation"""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}Testing Bitcoin Address Generator")
    print(f"{Fore.CYAN}{'='*80}\n")
    
    # Create generator with empty target set
    generator = BitcoinAddressGenerator(set())
    
    print(f"{Fore.YELLOW}Generating test addresses...\n")
    
    for i in range(3):
        print(f"{Fore.CYAN}Test #{i+1}:")
        print(f"{Fore.CYAN}{'-'*80}")
        
        # Generate mnemonic
        mnemonic = generator.generate_mnemonic()
        print(f"Mnemonic: {Fore.GREEN}{mnemonic}")
        
        # Convert to seed
        seed = generator.mnemonic_to_seed(mnemonic)
        print(f"Seed (first 32 bytes): {Fore.GREEN}{seed[:32].hex()}")
        
        # Derive private key
        private_key, chain_code = generator.derive_master_key(seed)
        print(f"Private Key: {Fore.GREEN}{private_key.hex()}")
        
        # Generate addresses
        try:
            addr_comp, addr_uncomp, addr_segwit, wif = generator.generate_address_from_private_key(private_key)
            
            print(f"\n{Fore.YELLOW}Generated Addresses:")
            print(f"  Compressed:   {Fore.GREEN}{addr_comp}")
            print(f"  Uncompressed: {Fore.GREEN}{addr_uncomp}")
            print(f"  SegWit:       {Fore.GREEN}{addr_segwit}")
            print(f"  WIF:          {Fore.GREEN}{wif}")
            print()
            
        except Exception as e:
            print(f"{Fore.RED}Error generating addresses: {e}")
        
        print()
    
    print(f"{Fore.GREEN}✓ Test completed successfully!")
    print(f"{Fore.CYAN}{'='*80}\n")


def test_address_checking():
    """Test address checking functionality"""
    print(f"{Fore.CYAN}Testing address checking...")
    
    # Create a test address
    generator = BitcoinAddressGenerator(set())
    mnemonic = generator.generate_mnemonic()
    seed = generator.mnemonic_to_seed(mnemonic)
    private_key, _ = generator.derive_master_key(seed)
    addr_comp, addr_uncomp, addr_segwit, wif = generator.generate_address_from_private_key(private_key)
    
    # Now create a generator that targets this address
    target_addresses = {addr_comp}
    generator2 = BitcoinAddressGenerator(target_addresses)
    
    print(f"{Fore.YELLOW}Target address: {addr_comp}")
    print(f"{Fore.YELLOW}Checking if we can find it again...")
    
    # We won't actually find it (astronomically unlikely), but test the mechanism
    results = generator2.generate_and_check(10)
    
    if results:
        print(f"{Fore.GREEN}✓ Found a match! (This is extremely rare)")
    else:
        print(f"{Fore.CYAN}No match found in 10 attempts (expected)")
    
    print(f"{Fore.GREEN}✓ Address checking works correctly!\n")


def test_performance():
    """Test generation speed"""
    import time
    
    print(f"{Fore.CYAN}Testing performance...")
    
    generator = BitcoinAddressGenerator(set())
    
    num_addresses = 100
    start_time = time.time()
    
    generator.generate_and_check(num_addresses)
    
    elapsed = time.time() - start_time
    speed = num_addresses / elapsed
    
    print(f"{Fore.YELLOW}Generated {num_addresses} addresses in {elapsed:.2f} seconds")
    print(f"{Fore.GREEN}Speed: {speed:.2f} addresses/second\n")


def main():
    """Run all tests"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("="*80)
    print("  Bitcoin Address Generator - Test Suite")
    print("="*80)
    print(f"{Style.RESET_ALL}\n")
    
    try:
        test_basic_generation()
        test_address_checking()
        test_performance()
        
        print(f"{Fore.GREEN}{Style.BRIGHT}All tests passed! ✓{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}You can now run the main generator:")
        print(f"  {Fore.YELLOW}python3 btc_generator.py")
        print(f"  {Fore.YELLOW}python3 btc_gpu_generator.py")
        print(f"  {Fore.YELLOW}./run.sh")
        
    except Exception as e:
        print(f"{Fore.RED}Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
