#!/usr/bin/env python3
"""
Optimization script to find best performance settings
"""

import time
import threading
from btc_generator import BitcoinAddressGenerator
from address_checker import AddressChecker

def benchmark_threads(thread_counts, duration=10):
    """Benchmark different thread counts"""
    print("Benchmarking different thread counts...")
    print(f"Duration: {duration} seconds each")
    print("-" * 50)
    
    results = {}
    
    for threads in thread_counts:
        print(f"Testing {threads} threads...")
        
        generator = BitcoinAddressGenerator(use_gpu=True)
        total_generated = 0
        start_time = time.time()
        
        def worker():
            nonlocal total_generated
            end_time = start_time + duration
            while time.time() < end_time:
                batch = generator.generate_address_batch(1000)
                total_generated += len(batch)
        
        # Start threads
        thread_list = []
        for _ in range(threads):
            t = threading.Thread(target=worker)
            thread_list.append(t)
            t.start()
        
        # Wait for completion
        for t in thread_list:
            t.join()
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed
        results[threads] = rate
        
        print(f"  {threads} threads: {rate:.2f} addr/sec")
    
    # Find best
    best_threads = max(results, key=results.get)
    best_rate = results[best_threads]
    
    print("-" * 50)
    print(f"Best performance: {best_threads} threads at {best_rate:.2f} addr/sec")
    return best_threads, best_rate

def benchmark_batch_sizes(batch_sizes, threads=8, duration=10):
    """Benchmark different batch sizes"""
    print(f"\nBenchmarking different batch sizes with {threads} threads...")
    print(f"Duration: {duration} seconds each")
    print("-" * 50)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        generator = BitcoinAddressGenerator(use_gpu=True)
        total_generated = 0
        start_time = time.time()
        
        def worker():
            nonlocal total_generated
            end_time = start_time + duration
            while time.time() < end_time:
                batch = generator.generate_address_batch(batch_size)
                total_generated += len(batch)
        
        # Start threads
        thread_list = []
        for _ in range(threads):
            t = threading.Thread(target=worker)
            thread_list.append(t)
            t.start()
        
        # Wait for completion
        for t in thread_list:
            t.join()
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed
        results[batch_size] = rate
        
        print(f"  Batch {batch_size}: {rate:.2f} addr/sec")
    
    # Find best
    best_batch = max(results, key=results.get)
    best_rate = results[best_batch]
    
    print("-" * 50)
    print(f"Best batch size: {best_batch} at {best_rate:.2f} addr/sec")
    return best_batch, best_rate

def main():
    print("Bitcoin Address Generator Optimization")
    print("=" * 50)
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16, 32]
    best_threads, best_thread_rate = benchmark_threads(thread_counts, duration=5)
    
    # Test different batch sizes with optimal thread count
    batch_sizes = [100, 500, 1000, 2000, 5000]
    best_batch, best_batch_rate = benchmark_batch_sizes(batch_sizes, best_threads, duration=5)
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Optimal threads: {best_threads}")
    print(f"Optimal batch size: {best_batch}")
    print(f"Best performance: {best_batch_rate:.2f} addresses/second")
    print("\nRecommended command:")
    print(f"python3 main.py --threads {best_threads} --batch-size {best_batch}")

if __name__ == "__main__":
    main()