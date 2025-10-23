"""
CUDA kernels for parallel Bitcoin address generation and hashing operations
Fallback to CPU implementation when CUDA is not available
"""

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("PyCUDA not available, using CPU fallback")

import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor

# CUDA kernel for parallel SHA256 hashing
cuda_sha256_kernel = """
#include <stdio.h>

__device__ void sha256_transform(unsigned int *state, const unsigned char *data) {
    // SHA256 constants
    const unsigned int k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    
    unsigned int w[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int t1, t2;
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i*4] << 24) | (data[i*4+1] << 16) | (data[i*4+2] << 8) | data[i*4+3];
    }
    
    for (int i = 16; i < 64; i++) {
        unsigned int s0 = ((w[i-15] >> 7) | (w[i-15] << 25)) ^ ((w[i-15] >> 18) | (w[i-15] << 14)) ^ (w[i-15] >> 3);
        unsigned int s1 = ((w[i-2] >> 17) | (w[i-2] << 15)) ^ ((w[i-2] >> 19) | (w[i-2] << 13)) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        unsigned int S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
        unsigned int ch = (e & f) ^ ((~e) & g);
        t1 = h + S1 + ch + k[i] + w[i];
        unsigned int S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + maj;
        
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__global__ void parallel_hash_kernel(unsigned char *input_data, unsigned char *output_hashes, int num_inputs, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_inputs) {
        unsigned int state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };
        
        unsigned char *input = input_data + idx * input_size;
        unsigned char *output = output_hashes + idx * 32;
        
        // Process input in 64-byte chunks
        int chunks = (input_size + 63) / 64;
        for (int chunk = 0; chunk < chunks; chunk++) {
            unsigned char block[64];
            int block_size = min(64, input_size - chunk * 64);
            
            for (int i = 0; i < block_size; i++) {
                block[i] = input[chunk * 64 + i];
            }
            
            // Padding for last block
            if (chunk == chunks - 1) {
                if (block_size < 64) {
                    block[block_size] = 0x80;
                    for (int i = block_size + 1; i < 64; i++) {
                        block[i] = 0;
                    }
                }
                
                // Add length in bits to last 8 bytes
                unsigned long long bit_len = input_size * 8;
                for (int i = 0; i < 8; i++) {
                    block[56 + i] = (bit_len >> (56 - i * 8)) & 0xff;
                }
            }
            
            sha256_transform(state, block);
        }
        
        // Convert state to output bytes
        for (int i = 0; i < 8; i++) {
            output[i*4] = (state[i] >> 24) & 0xff;
            output[i*4+1] = (state[i] >> 16) & 0xff;
            output[i*4+2] = (state[i] >> 8) & 0xff;
            output[i*4+3] = state[i] & 0xff;
        }
    }
}
"""

class CudaHasher:
    def __init__(self):
        if CUDA_AVAILABLE:
            try:
                self.mod = SourceModule(cuda_sha256_kernel)
                self.hash_kernel = self.mod.get_function("parallel_hash_kernel")
                self.use_cuda = True
                print("CUDA hasher initialized successfully")
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                self.use_cuda = False
        else:
            self.use_cuda = False
            print("Using CPU fallback for hashing")
    
    def parallel_hash(self, input_data_list):
        """Perform parallel SHA256 hashing on GPU or CPU"""
        if self.use_cuda and CUDA_AVAILABLE:
            return self._gpu_hash(input_data_list)
        else:
            return self._cpu_hash(input_data_list)
    
    def _gpu_hash(self, input_data_list):
        """GPU implementation"""
        num_inputs = len(input_data_list)
        max_input_size = max(len(data) for data in input_data_list)
        
        # Prepare input data
        input_array = np.zeros((num_inputs, max_input_size), dtype=np.uint8)
        for i, data in enumerate(input_data_list):
            input_array[i, :len(data)] = np.frombuffer(data, dtype=np.uint8)
        
        # Allocate GPU memory
        input_gpu = cuda.mem_alloc(input_array.nbytes)
        output_gpu = cuda.mem_alloc(num_inputs * 32)  # 32 bytes per SHA256 hash
        
        # Copy data to GPU
        cuda.memcpy_htod(input_gpu, input_array)
        
        # Launch kernel
        block_size = 256
        grid_size = (num_inputs + block_size - 1) // block_size
        
        self.hash_kernel(
            input_gpu, output_gpu, np.int32(num_inputs), np.int32(max_input_size),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        # Copy results back
        output_array = np.zeros(num_inputs * 32, dtype=np.uint8)
        cuda.memcpy_dtoh(output_array, output_gpu)
        
        # Free GPU memory
        input_gpu.free()
        output_gpu.free()
        
        # Reshape and return results
        return output_array.reshape((num_inputs, 32))
    
    def _cpu_hash(self, input_data_list):
        """CPU fallback implementation using multithreading"""
        def hash_single(data):
            return np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8)
        
        # Use ThreadPoolExecutor for parallel CPU hashing
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(hash_single, input_data_list))
        
        return np.array(results)