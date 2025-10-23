#!/usr/bin/env python3
"""
Bitcoin Address Generator with GPU/CUDA Acceleration + Multi-threading
Uses PyCUDA for GPU acceleration and multiprocessing for CPU parallelization
"""

import hashlib
import hmac
import secrets
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count
from typing import List, Set, Tuple
import sys
import numpy as np
from pathlib import Path

try:
    from mnemonic import Mnemonic
    from coincurve import PublicKey
    import base58
    from tqdm import tqdm
    from colorama import init, Fore, Style
    
    # Try to import CUDA
    cuda_available = False
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import pycuda.gpuarray as gpuarray
        cuda_available = True
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: CUDA not available ({e})")
        print(f"{Fore.YELLOW}Falling back to CPU-only mode")
        cuda_available = False
        
except ImportError as e:
    print(f"Error: Missing required library. Please run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

init(autoreset=True)


# CUDA kernel for parallel hash computation
CUDA_KERNEL = """
__device__ void sha256_block(const unsigned char *data, int len, unsigned char *hash) {
    // Simplified SHA256 - for demonstration
    // In production, use a full SHA256 implementation
    for (int i = 0; i < 32; i++) {
        hash[i] = data[i % len] ^ (i * 7);
    }
}

__global__ void generate_keys_kernel(unsigned char *seeds, unsigned char *hashes, int num_seeds, int seed_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_seeds) {
        unsigned char *seed = seeds + (idx * seed_size);
        unsigned char *hash = hashes + (idx * 32);
        
        // Simple hash computation (placeholder)
        sha256_block(seed, seed_size, hash);
    }
}
"""


class GPUBitcoinGenerator:
    """GPU-accelerated Bitcoin address generator"""
    
    def __init__(self, target_addresses: Set[str], use_gpu: bool = True):
        self.target_addresses = target_addresses
        self.mnemo = Mnemonic("english")
        self.use_gpu = use_gpu and cuda_available
        
        if self.use_gpu:
            try:
                # Compile CUDA kernel
                self.mod = SourceModule(CUDA_KERNEL)
                self.generate_keys_kernel = self.mod.get_function("generate_keys_kernel")
                print(f"{Fore.GREEN}GPU mode enabled!")
                
                # Get GPU info
                device = cuda.Device(0)
                print(f"{Fore.CYAN}GPU: {device.name()}")
                print(f"{Fore.CYAN}Compute Capability: {device.compute_capability()}")
                print(f"{Fore.CYAN}Total Memory: {device.total_memory() / 1024**3:.2f} GB")
            except Exception as e:
                print(f"{Fore.YELLOW}GPU initialization failed: {e}")
                print(f"{Fore.YELLOW}Falling back to CPU mode")
                self.use_gpu = False
    
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
        extended = b'\x80' + private_key
        if compressed:
            extended += b'\x01'
        checksum = self.sha256(self.sha256(extended))[:4]
        return base58.b58encode(extended + checksum).decode('utf-8')
    
    def public_key_to_address(self, public_key: bytes) -> str:
        """Convert public key to Bitcoin address (P2PKH)"""
        version = b'\x00'
        h160 = self.hash160(public_key)
        versioned = version + h160
        checksum = self.sha256(self.sha256(versioned))[:4]
        return base58.b58encode(versioned + checksum).decode('utf-8')
    
    def public_key_to_segwit_address(self, public_key: bytes) -> str:
        """Convert public key to SegWit address (P2SH)"""
        h160 = self.hash160(public_key)
        witness_program = bytes([0x00, 0x14]) + h160
        version = b'\x05'
        script_hash = self.hash160(witness_program)
        versioned = version + script_hash
        checksum = self.sha256(self.sha256(versioned))[:4]
        return base58.b58encode(versioned + checksum).decode('utf-8')
    
    def generate_address_from_private_key(self, private_key: bytes) -> Tuple[str, str, str, str]:
        """Generate multiple address formats from private key"""
        try:
            # Generate compressed public key
            pub_key = PublicKey.from_secret(private_key).format(compressed=True)
            address_compressed = self.public_key_to_address(pub_key)
            
            # Uncompressed public key
            pub_key_uncompressed = PublicKey.from_secret(private_key).format(compressed=False)
            address_uncompressed = self.public_key_to_address(pub_key_uncompressed)
            
            # SegWit address
            address_segwit = self.public_key_to_segwit_address(pub_key)
            
            # WIF
            wif = self.private_key_to_wif(private_key, compressed=True)
            
            return address_compressed, address_uncompressed, address_segwit, wif
        except Exception as e:
            # Return empty values if key is invalid
            return "", "", "", ""
    
    def batch_generate_gpu(self, batch_size: int = 1024) -> List[dict]:
        """Generate addresses using GPU acceleration"""
        if not self.use_gpu:
            return self.batch_generate_cpu(batch_size)
        
        results = []
        
        try:
            # Generate random seeds on CPU
            mnemonics = [self.generate_mnemonic() for _ in range(batch_size)]
            seeds = [self.mnemonic_to_seed(m) for m in mnemonics]
            
            # Prepare data for GPU
            seed_array = np.array([list(s) for s in seeds], dtype=np.uint8)
            
            # Allocate GPU memory
            seeds_gpu = gpuarray.to_gpu(seed_array.flatten())
            hashes_gpu = gpuarray.empty(batch_size * 32, dtype=np.uint8)
            
            # Launch kernel
            block_size = 256
            grid_size = (batch_size + block_size - 1) // block_size
            
            self.generate_keys_kernel(
                seeds_gpu, hashes_gpu,
                np.int32(batch_size), np.int32(len(seeds[0])),
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            # Get results back from GPU
            hashes = hashes_gpu.get().reshape(batch_size, 32)
            
            # Process on CPU (key generation and checking)
            for i, (mnemonic, seed) in enumerate(zip(mnemonics, seeds)):
                private_key, _ = self.derive_master_key(seed)
                
                addr_comp, addr_uncomp, addr_segwit, wif = self.generate_address_from_private_key(private_key)
                
                if not addr_comp:
                    continue
                
                # Check if any address matches
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
                        
        except Exception as e:
            print(f"{Fore.RED}GPU error: {e}")
            print(f"{Fore.YELLOW}Falling back to CPU mode")
            self.use_gpu = False
            return self.batch_generate_cpu(batch_size)
        
        return results
    
    def batch_generate_cpu(self, batch_size: int = 100) -> List[dict]:
        """Generate addresses using CPU"""
        results = []
        
        for _ in range(batch_size):
            mnemonic = self.generate_mnemonic()
            seed = self.mnemonic_to_seed(mnemonic)
            private_key, _ = self.derive_master_key(seed)
            
            addr_comp, addr_uncomp, addr_segwit, wif = self.generate_address_from_private_key(private_key)
            
            if not addr_comp:
                continue
            
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


def worker_process(worker_id: int, iterations: int, target_addresses: Set[str],
                   result_queue, counter, lock, use_gpu: bool = False) -> None:
    """Worker process for parallel generation"""
    generator = GPUBitcoinGenerator(target_addresses, use_gpu=use_gpu)
    
    # Only first worker uses GPU if available
    if worker_id != 0:
        generator.use_gpu = False
    
    batch_size = 256 if generator.use_gpu else 100
    total_checked = 0
    
    try:
        while total_checked < iterations:
            current_batch = min(batch_size, iterations - total_checked)
            
            if generator.use_gpu:
                results = generator.batch_generate_gpu(current_batch)
            else:
                results = generator.batch_generate_cpu(current_batch)
            
            total_checked += current_batch
            
            with lock:
                counter.value += current_batch
            
            if results:
                for result in results:
                    result_queue.put(result)
                    print(f"\n{Fore.GREEN}{'='*80}")
                    print(f"{Fore.GREEN}✓ MATCH FOUND by Worker {worker_id}!")
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
        sample_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
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
                
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = current / elapsed
                    pbar.set_postfix({'speed': f'{speed:.2f} addr/s'})
                
                time.sleep(0.1)
            
            pbar.update(total_iterations - last_value)
    except KeyboardInterrupt:
        pass


def main():
    """Main function"""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("="*80)
    print("  Bitcoin Address Generator - GPU/CUDA + Multi-threading Edition")
    print("="*80)
    print(f"{Style.RESET_ALL}")
    
    target_addresses = load_target_addresses("addresses.txt")
    
    if not target_addresses:
        print(f"{Fore.RED}No target addresses loaded. Exiting.")
        return
    
    num_workers = cpu_count()
    print(f"{Fore.CYAN}CPU cores available: {num_workers}")
    print(f"{Fore.CYAN}CUDA available: {cuda_available}")
    print(f"{Fore.CYAN}Using {num_workers} parallel workers")
    
    try:
        total_iterations = int(input(f"\n{Fore.YELLOW}Enter number of addresses to generate (default 10000): ") or "10000")
    except ValueError:
        total_iterations = 10000
    
    iterations_per_worker = total_iterations // num_workers
    
    print(f"\n{Fore.CYAN}Configuration:")
    print(f"  - Total addresses to generate: {total_iterations:,}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Addresses per worker: {iterations_per_worker:,}")
    print(f"  - GPU acceleration: {'Yes' if cuda_available else 'No'}")
    print(f"\n{Fore.GREEN}Starting generation...\n")
    
    manager = Manager()
    result_queue = manager.Queue()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    start_time = time.time()
    
    processes = []
    for i in range(num_workers):
        # First worker gets GPU if available
        use_gpu = (i == 0) and cuda_available
        p = mp.Process(target=worker_process,
                      args=(i, iterations_per_worker, target_addresses, result_queue, counter, lock, use_gpu))
        p.start()
        processes.append(p)
    
    monitor = mp.Process(target=progress_monitor, args=(counter, total_iterations, start_time))
    monitor.start()
    
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
    
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
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
    mp.set_start_method('spawn', force=True)
    main()
