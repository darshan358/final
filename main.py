"""
Main Bitcoin Address Generator with GPU acceleration and parallel processing
"""

import time
import threading
import signal
import sys
import os
from typing import List, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from btc_generator import BitcoinAddressGenerator
from address_checker import AddressChecker

class BitcoinAddressMiner:
    def __init__(self, use_gpu=True, num_threads=8, batch_size=1000):
        self.generator = BitcoinAddressGenerator(use_gpu=use_gpu)
        self.checker = AddressChecker()
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.running = False
        self.stats = {
            'addresses_generated': 0,
            'addresses_checked': 0,
            'matches_found': 0,
            'start_time': 0,
            'runtime': 0
        }
        self.stats_lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def update_stats(self, generated=0, checked=0, matches=0):
        """Update statistics thread-safely"""
        with self.stats_lock:
            self.stats['addresses_generated'] += generated
            self.stats['addresses_checked'] += checked
            self.stats['matches_found'] += matches
            self.stats['runtime'] = time.time() - self.stats['start_time']
    
    def print_stats(self):
        """Print current statistics"""
        with self.stats_lock:
            runtime = self.stats['runtime']
            generated = self.stats['addresses_generated']
            checked = self.stats['addresses_checked']
            matches = self.stats['matches_found']
            
            rate_gen = generated / runtime if runtime > 0 else 0
            rate_check = checked / runtime if runtime > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"MINING STATISTICS")
            print(f"{'='*60}")
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Addresses Generated: {generated:,}")
            print(f"Addresses Checked: {checked:,}")
            print(f"Matches Found: {matches}")
            print(f"Generation Rate: {rate_gen:.2f} addr/sec")
            print(f"Check Rate: {rate_check:.2f} addr/sec")
            print(f"Target Addresses: {len(self.checker.target_addresses):,}")
            print(f"{'='*60}")
    
    def worker_thread(self, thread_id: int):
        """Worker thread for generating and checking addresses"""
        print(f"Worker thread {thread_id} started")
        
        while self.running:
            try:
                # Generate batch of addresses
                batch = self.generator.generate_address_batch(self.batch_size)
                self.update_stats(generated=len(batch))
                
                # Check addresses
                matches = self.checker.check_addresses(batch)
                self.update_stats(checked=len(batch), matches=len(matches))
                
                # Save matches if found
                if matches:
                    self.checker.save_matches()
                
            except Exception as e:
                print(f"Error in worker thread {thread_id}: {e}")
                time.sleep(1)  # Brief pause before retrying
        
        print(f"Worker thread {thread_id} stopped")
    
    def stats_thread(self):
        """Thread for periodic statistics display"""
        while self.running:
            time.sleep(10)  # Update every 10 seconds
            if self.running:
                self.print_stats()
    
    def run_mining(self, duration: int = None):
        """Run the mining operation"""
        print("Starting Bitcoin Address Mining...")
        print(f"Threads: {self.num_threads}")
        print(f"Batch Size: {self.batch_size}")
        print(f"GPU Acceleration: {'Enabled' if self.generator.use_gpu else 'Disabled'}")
        print(f"Target Addresses: {len(self.checker.target_addresses):,}")
        
        if len(self.checker.target_addresses) == 0:
            print("Warning: No target addresses loaded. Add addresses to addresses.txt")
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker_thread, args=(i,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Start statistics thread
        stats_thread = threading.Thread(target=self.stats_thread)
        stats_thread.daemon = True
        stats_thread.start()
        
        try:
            # Run for specified duration or until interrupted
            if duration:
                print(f"Running for {duration} seconds...")
                time.sleep(duration)
                self.running = False
            else:
                print("Running indefinitely. Press Ctrl+C to stop...")
                while self.running:
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            self.running = False
        
        # Wait for threads to finish
        print("Waiting for threads to finish...")
        for thread in threads:
            thread.join(timeout=5)
        
        # Final statistics
        self.print_stats()
        
        # Save any remaining matches
        self.checker.save_matches()
        
        print("Mining stopped.")
    
    def generate_sample_addresses(self, count: int = 10):
        """Generate sample addresses for testing"""
        print(f"Generating {count} sample addresses...")
        
        addresses = self.generator.generate_address_batch(count)
        
        print(f"\n{'='*80}")
        print("SAMPLE GENERATED ADDRESSES")
        print(f"{'='*80}")
        
        for i, addr in enumerate(addresses, 1):
            print(f"\nAddress {i}:")
            print(f"  Mnemonic: {addr['mnemonic']}")
            print(f"  Private Key: {addr['private_key_wif']}")
            print(f"  P2PKH: {addr['p2pkh_address']}")
            print(f"  P2SH: {addr['p2sh_address']}")
            print(f"  Bech32: {addr['bech32_address']}")
        
        print(f"\n{'='*80}")
    
    def benchmark(self, duration: int = 60):
        """Run benchmark test"""
        print(f"Running benchmark for {duration} seconds...")
        
        start_time = time.time()
        total_generated = 0
        
        while time.time() - start_time < duration:
            batch = self.generator.generate_address_batch(self.batch_size)
            total_generated += len(batch)
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed
        
        print(f"\nBenchmark Results:")
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Addresses Generated: {total_generated:,}")
        print(f"Rate: {rate:.2f} addresses/second")
        print(f"GPU Acceleration: {'Enabled' if self.generator.use_gpu else 'Disabled'}")

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Address Generator and Checker")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size (default: 1000)")
    parser.add_argument("--duration", type=int, help="Run duration in seconds (default: indefinite)")
    parser.add_argument("--sample", type=int, help="Generate sample addresses and exit")
    parser.add_argument("--benchmark", type=int, help="Run benchmark for specified seconds")
    parser.add_argument("--add-address", type=str, help="Add target address to addresses.txt")
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = BitcoinAddressMiner(
        use_gpu=not args.no_gpu,
        num_threads=args.threads,
        batch_size=args.batch_size
    )
    
    # Handle different modes
    if args.add_address:
        miner.checker.add_target_address(args.add_address)
        print(f"Added address: {args.add_address}")
        return
    
    if args.sample:
        miner.generate_sample_addresses(args.sample)
        return
    
    if args.benchmark:
        miner.benchmark(args.benchmark)
        return
    
    # Run mining
    miner.run_mining(args.duration)

if __name__ == "__main__":
    main()