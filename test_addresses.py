#!/usr/bin/env python3
"""
Test script to verify address generation and validation
"""

from btc_generator import BitcoinAddressGenerator
from address_checker import AddressChecker
import hashlib
import base58

def validate_bitcoin_address(address):
    """Validate Bitcoin address format"""
    if address.startswith('1') or address.startswith('3'):
        # Base58 address
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 25:
                return False
            # Check checksum
            payload = decoded[:-4]
            checksum = decoded[-4:]
            hash_result = hashlib.sha256(hashlib.sha256(payload).digest()).digest()
            return checksum == hash_result[:4]
        except:
            return False
    elif address.startswith('bc1'):
        # Bech32 address - simplified validation
        return len(address) >= 14 and len(address) <= 74
    return False

def test_address_generation():
    """Test address generation functionality"""
    print("Testing address generation...")
    
    generator = BitcoinAddressGenerator(use_gpu=False)
    
    # Generate test addresses
    addresses = generator.generate_address_batch(5)
    
    print(f"Generated {len(addresses)} addresses")
    
    for i, addr in enumerate(addresses, 1):
        print(f"\nAddress {i}:")
        print(f"  Mnemonic: {addr['mnemonic']}")
        print(f"  P2PKH: {addr['p2pkh_address']}")
        print(f"  P2SH: {addr['p2sh_address']}")
        print(f"  Bech32: {addr['bech32_address']}")
        
        # Validate addresses
        p2pkh_valid = validate_bitcoin_address(addr['p2pkh_address'])
        p2sh_valid = validate_bitcoin_address(addr['p2sh_address'])
        bech32_valid = validate_bitcoin_address(addr['bech32_address'])
        
        print(f"  P2PKH Valid: {p2pkh_valid}")
        print(f"  P2SH Valid: {p2sh_valid}")
        print(f"  Bech32 Valid: {bech32_valid}")
        
        if not all([p2pkh_valid, p2sh_valid, bech32_valid]):
            print("  ❌ VALIDATION FAILED")
            return False
        else:
            print("  ✅ All addresses valid")
    
    return True

def test_address_checking():
    """Test address checking functionality"""
    print("\nTesting address checking...")
    
    checker = AddressChecker()
    
    # Create test addresses
    test_addresses = [
        {
            'mnemonic': 'test mnemonic phrase',
            'private_key_wif': 'test_key',
            'p2pkh_address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',  # Genesis block address
            'p2sh_address': '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
            'bech32_address': 'bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4'
        }
    ]
    
    matches = checker.check_addresses(test_addresses)
    
    if matches:
        print(f"Found {len(matches)} matches (expected if genesis address is in addresses.txt)")
        for match in matches:
            print(f"  Matched: {match['matched_address']}")
    else:
        print("No matches found (expected if genesis address not in addresses.txt)")
    
    return True

def test_mnemonic_validation():
    """Test mnemonic phrase validation"""
    print("\nTesting mnemonic validation...")
    
    generator = BitcoinAddressGenerator(use_gpu=False)
    
    # Generate multiple mnemonics and check they're valid
    for i in range(5):
        mnemonic = generator.generate_random_mnemonic()
        print(f"Mnemonic {i+1}: {mnemonic}")
        
        # Check word count
        words = mnemonic.split()
        if len(words) not in [12, 15, 18, 21, 24]:
            print(f"  ❌ Invalid word count: {len(words)}")
            return False
        
        # Validate mnemonic
        if not generator.mnemo.check(mnemonic):
            print(f"  ❌ Invalid mnemonic")
            return False
        
        print(f"  ✅ Valid mnemonic ({len(words)} words)")
    
    return True

def main():
    print("Bitcoin Address Generator Test Suite")
    print("=" * 50)
    
    tests = [
        ("Address Generation", test_address_generation),
        ("Address Checking", test_address_checking),
        ("Mnemonic Validation", test_mnemonic_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)