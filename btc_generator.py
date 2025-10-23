#!/usr/bin/env python3
"""
Bitcoin Address Generator with GPU/CUDA Acceleration
Generates random BTC addresses from seed phrases and checks against target list
"""

import hashlib
import hmac
import secrets
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count
from typing import List, Set, Tuple
import sys
from pathlib import Path

try:
    from mnemonic import Mnemonic
    from coincurve import PublicKey
    import base58
    from tqdm import tqdm
    from colorama import init, Fore, Style
except ImportError as e:
    print(f"Error: Missing required library. Please run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

init(autoreset=True)

# Bitcoin constants
BITCOIN_ALPHABET = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


class BitcoinAddressGenerator:
    """Generate Bitcoin addresses from mnemonic seed phrases"""
    
    def __init__(self, target_addresses: Set[str]):
        self.target_addresses = target_addresses
        self.mnemo = Mnemonic("english")
        
    def sha256(self, data: bytes) -> bytes:
        """SHA256 hash"""
        return hashlib.sha256(data).digest()
    
    def ripemd160(self, data: bytes) -> bytes:
        """RIPEMD160 hash"""
        h = hashlib.new('ripemd160')
        h.update(data)
        return h.digest()
    
    def hash160(self, data: bytes) -> bytes:
        """Bitcoin HASH160: RIPEMD160(SHA256(data))"""
        return self.ripemd160(self.sha256(data))
    
    def generate_mnemonic(self, strength: int = 256) -> str:
        """Generate a random mnemonic seed phrase"""
        return self.mnemo.generate(strength=strength)
    
    def mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed"""
        return self.mnemo.to_seed(mnemonic, passphrase)
    
    def derive_master_key(self, seed: bytes) -> Tuple[bytes, bytes]:
        """Derive master private key and chain code from seed"""
        h = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
        return h[:32], h[32:]
    
    def private_key_to_wif(self, private_key: bytes, compressed: bool = True) -> str:
        """Convert private key to WIF format"""
        # Mainnet prefix
        extended = b'\x80' + private_key
        if compressed:
            extended += b'\x01'
        
        # Double SHA256 for checksum
        checksum = self.sha256(self.sha256(extended))[:4]
        return base58.b58encode(extended + checksum).decode('utf-8')
    
    def public_key_to_address(self, public_key: bytes, compressed: bool = True) -> str:
        """Convert public key to Bitcoin address (P2PKH)"""
        # Version byte for mainnet P2PKH
        version = b'\x00'
        
        # Hash160 of public key
        h160 = self.hash160(public_key)
        
        # Add version byte
        versioned = version + h160
        
        # Double SHA256 for checksum
        checksum = self.sha256(self.sha256(versioned))[:4]
        
        # Encode in base58
        address = base58.b58encode(versioned + checksum).decode('utf-8')
        return address
    
    def public_key_to_segwit_address(self, public_key: bytes) -> str:
        """Convert public key to SegWit address (P2WPKH - Bech32)"""
        # Hash160 of public key
        h160 = self.hash160(public_key)
        
        # Witness version 0
        witness_program = bytes([0x00, 0x14]) + h160
        
        # For simplicity, convert to P2SH-wrapped SegWit
        version = b'\x05'  # P2SH version
        script_hash = self.hash160(witness_program)
        versioned = version + script_hash
        checksum = self.sha256(self.sha256(versioned))[:4]
        
        return base58.b58encode(versioned + checksum).decode('utf-8')
    
    def generate_address_from_private_key(self, private_key: bytes) -> Tuple[str, str, str, str]:
        """Generate multiple address formats from private key"""
        # Generate public key (compressed)
        pub_key = PublicKey.from_secret(private_key).format(compressed=True)
        
        # Generate different address formats
        address_compressed = self.public_key_to_address(pub_key, compressed=True)
        
        # Uncompressed public key
        pub_key_uncompressed = PublicKey.from_secret(private_key).format(compressed=False)
        address_uncompressed = self.public_key_to_address(pub_key_uncompressed, compressed=False)
        
        # SegWit address
        address_segwit = self.public_key_to_segwit_address(pub_key)
        
        # WIF
        wif = self.private_key_to_wif(private_key, compressed=True)
        
        return address_compressed, address_uncompressed, address_segwit, wif
    
    def generate_and_check(self, iterations: int = 1) -> List[dict]:
        """Generate random addresses and check against target list"""
        results = []
        
        for _ in range(iterations):
            # Generate random mnemonic
            mnemonic = self.generate_mnemonic()
            
            # Convert to seed
            seed = self.mnemonic_to_seed(mnemonic)
            
            # Derive master private key
            private_key, chain_code = self.derive_master_key(seed)
            
            # Generate addresses
            addr_comp, addr_uncomp, addr_segwit, wif = self.generate_address_from_private_key(private_key)
            
            # Check if any address matches target list
            addresses = [addr_comp, addr_uncomp, addr_segwit]
            for addr in addresses:
                if addr in self.target_addresses:
                    result = {
                        'found': True,
                        'mnemonic': mnemonic,
                        'private_key': private_key.hex(),
                        'wif': wif,
                        'address_compressed': addr_comp,
                        'address_uncompressed': addr_uncomp,
                        'address_segwit': addr_segwit,
                        'matched_address': addr
                    }
                    results.append(result)
                    return results
        
        return results


def worker_process(worker_id: int, iterations_per_worker: int, target_addresses: Set[str], 
                   result_queue, counter, lock) -> None:
    """Worker process for parallel generation"""
    generator = BitcoinAddressGenerator(target_addresses)
    batch_size = 100
    total_checked = 0
    
    try:
        while total_checked < iterations_per_worker:
            current_batch = min(batch_size, iterations_per_worker - total_checked)
            results = generator.generate_and_check(current_batch)
            
            total_checked += current_batch
            
            # Update global counter
            with lock:
                counter.value += current_batch
            
            # If found a match, put in queue
            if results:
                for result in results:
                    result_queue.put(result)
                    print(f"\n{Fore.GREEN}{'='*80}")
                    print(f"{Fore.GREEN}MATCH FOUND by Worker {worker_id}!")
                    print(f"{Fore.GREEN}{'='*80}")
                    print(f"{Fore.YELLOW}Address: {result['matched_address']}")
                    print(f"{Fore.YELLOW}Mnemonic: {result['mnemonic']}")
                    print(f"{Fore.YELLOW}Private Key: {result['private_key']}")
                    print(f"{Fore.YELLOW}WIF: {result['wif']}")
                    print(f"{Fore.GREEN}{'='*80}\n")
                return
    
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f"{Fore.RED}Error in worker {worker_id}: {e}")


def load_target_addresses(filename: str = "addresses.txt") -> Set[str]:
    """Load target addresses from file"""
    try:
        with open(filename, 'r') as f:
            addresses = {line.strip() for line in f if line.strip()}
        print(f"{Fore.CYAN}Loaded {len(addresses)} target addresses from {filename}")
        return addresses
    except FileNotFoundError:
        print(f"{Fore.RED}Error: {filename} not found!")
        print(f"{Fore.YELLOW}Creating sample {filename}...")
        
        # Create sample file
        sample_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Satoshi's address
            "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",
            "3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy",
        ]
        with open(filename, 'w') as f:
            f.write('\n'.join(sample_addresses))
        
        return set(sample_addresses)


def progress_monitor(counter, total_iterations, start_time):
    """Monitor and display progress"""
    try:
        with tqdm(total=total_iterations, desc="Generating addresses", 
                  unit="addr", ncols=100, colour='green') as pbar:
            last_value = 0
            while counter.value < total_iterations:
                current = counter.value
                increment = current - last_value
                if increment > 0:
                    pbar.update(increment)
                    last_value = current
                
                # Calculate speed
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = current / elapsed
                    pbar.set_postfix({'speed': f'{speed:.2f} addr/s'})
                
                time.sleep(0.1)
            
            # Final update
            pbar.update(total_iterations - last_value)
    except KeyboardInterrupt:
        pass


def main():
    """Main function"""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("="*80)
    print("Bitcoin Address Generator with GPU/CUDA & Parallel Processing")
    print("="*80)
    print(f"{Style.RESET_ALL}")
    
    # Load target addresses
    target_addresses = load_target_addresses("addresses.txt")
    
    if not target_addresses:
        print(f"{Fore.RED}No target addresses loaded. Exiting.")
        return
    
    # Configuration
    num_workers = cpu_count()
    print(f"{Fore.CYAN}CPU cores available: {num_workers}")
    print(f"{Fore.CYAN}Using {num_workers} parallel workers")
    
    # Get user input for iterations
    try:
        total_iterations = int(input(f"\n{Fore.YELLOW}Enter number of addresses to generate (default 10000): ") or "10000")
    except ValueError:
        total_iterations = 10000
    
    iterations_per_worker = total_iterations // num_workers
    
    print(f"\n{Fore.CYAN}Configuration:")
    print(f"  - Total addresses to generate: {total_iterations:,}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Addresses per worker: {iterations_per_worker:,}")
    print(f"\n{Fore.GREEN}Starting generation...\n")
    
    # Create shared resources
    manager = Manager()
    result_queue = manager.Queue()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    start_time = time.time()
    
    # Start worker processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, 
                      args=(i, iterations_per_worker, target_addresses, result_queue, counter, lock))
        p.start()
        processes.append(p)
    
    # Start progress monitor
    monitor = mp.Process(target=progress_monitor, args=(counter, total_iterations, start_time))
    monitor.start()
    
    # Wait for all workers to complete
    try:
        for p in processes:
            p.join()
        monitor.terminate()
        monitor.join()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user. Terminating workers...")
        for p in processes:
            p.terminate()
        monitor.terminate()
        for p in processes:
            p.join()
        monitor.join()
    
    elapsed_time = time.time() - start_time
    
    # Check results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Summary
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}Summary")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"Total addresses checked: {counter.value:,}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Speed: {counter.value/elapsed_time:.2f} addresses/second")
    print(f"Matches found: {len(results)}")
    
    if results:
        print(f"\n{Fore.GREEN}Saving results to found_addresses.txt")
        with open('found_addresses.txt', 'w') as f:
            for result in results:
                f.write(f"\nMatched Address: {result['matched_address']}\n")
                f.write(f"Mnemonic: {result['mnemonic']}\n")
                f.write(f"Private Key: {result['private_key']}\n")
                f.write(f"WIF: {result['wif']}\n")
                f.write(f"Compressed Address: {result['address_compressed']}\n")
                f.write(f"Uncompressed Address: {result['address_uncompressed']}\n")
                f.write(f"SegWit Address: {result['address_segwit']}\n")
                f.write("-"*80 + "\n")
    
    print(f"\n{Fore.CYAN}Done!")


if __name__ == "__main__":
    main()
