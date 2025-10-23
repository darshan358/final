"""
Bitcoin address generator using seed phrases with GPU acceleration
"""

import hashlib
import hmac
import struct
import base58
from mnemonic import Mnemonic
from ecdsa import SigningKey, SECP256k1
from ecdsa.util import string_to_number
import secrets
import numpy as np
from typing import List, Tuple, Optional
import concurrent.futures
import threading
try:
    from cuda_kernels import CudaHasher
except ImportError:
    print("CUDA kernels not available, using CPU-only mode")
    CudaHasher = None

class BitcoinAddressGenerator:
    def __init__(self, use_gpu=True):
        self.mnemo = Mnemonic("english")
        self.use_gpu = use_gpu and CudaHasher is not None
        if self.use_gpu:
            try:
                self.cuda_hasher = CudaHasher()
                print("GPU acceleration enabled")
            except Exception as e:
                print(f"GPU acceleration failed, falling back to CPU: {e}")
                self.use_gpu = False
                self.cuda_hasher = None
        else:
            self.cuda_hasher = None
            print("Using CPU-only mode")
    
    def generate_random_mnemonic(self, strength: int = 256) -> str:
        """Generate a random mnemonic seed phrase"""
        entropy = secrets.randbits(strength)
        entropy_bytes = entropy.to_bytes(strength // 8, byteorder='big')
        return self.mnemo.to_mnemonic(entropy_bytes)
    
    def mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed using PBKDF2"""
        mnemonic_bytes = mnemonic.encode('utf-8')
        salt = ("mnemonic" + passphrase).encode('utf-8')
        return hashlib.pbkdf2_hmac('sha512', mnemonic_bytes, salt, 2048)
    
    def derive_master_key(self, seed: bytes) -> Tuple[bytes, bytes]:
        """Derive master private key and chain code from seed"""
        hmac_result = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
        master_private_key = hmac_result[:32]
        master_chain_code = hmac_result[32:]
        return master_private_key, master_chain_code
    
    def derive_child_key(self, parent_key: bytes, parent_chain: bytes, index: int) -> Tuple[bytes, bytes]:
        """Derive child key using BIP32 derivation"""
        if index >= 2**31:  # Hardened derivation
            data = b'\x00' + parent_key + struct.pack('>I', index)
        else:  # Non-hardened derivation
            # Get public key from private key
            sk = SigningKey.from_string(parent_key, curve=SECP256k1)
            public_key = sk.get_verifying_key().to_string("compressed")
            data = public_key + struct.pack('>I', index)
        
        hmac_result = hmac.new(parent_chain, data, hashlib.sha512).digest()
        child_key = hmac_result[:32]
        child_chain = hmac_result[32:]
        
        # Add parent key to child key (modulo curve order)
        parent_int = string_to_number(parent_key)
        child_int = string_to_number(child_key)
        final_key_int = (parent_int + child_int) % SECP256k1.order
        final_key = final_key_int.to_bytes(32, byteorder='big')
        
        return final_key, child_chain
    
    def derive_address_key(self, seed: bytes, path: str = "m/44'/0'/0'/0/0") -> bytes:
        """Derive private key for specific BIP44 path"""
        master_key, master_chain = self.derive_master_key(seed)
        
        # Parse derivation path
        path_elements = path.split('/')[1:]  # Skip 'm'
        current_key = master_key
        current_chain = master_chain
        
        for element in path_elements:
            if element.endswith("'"):
                index = int(element[:-1]) + 2**31  # Hardened
            else:
                index = int(element)
            current_key, current_chain = self.derive_child_key(current_key, current_chain, index)
        
        return current_key
    
    def private_key_to_wif(self, private_key: bytes, compressed: bool = True) -> str:
        """Convert private key to Wallet Import Format"""
        extended_key = b'\x80' + private_key
        if compressed:
            extended_key += b'\x01'
        
        # Double SHA256
        hash1 = hashlib.sha256(extended_key).digest()
        hash2 = hashlib.sha256(hash1).digest()
        checksum = hash2[:4]
        
        return base58.b58encode(extended_key + checksum).decode()
    
    def private_key_to_public_key(self, private_key: bytes, compressed: bool = True) -> bytes:
        """Convert private key to public key"""
        sk = SigningKey.from_string(private_key, curve=SECP256k1)
        vk = sk.get_verifying_key()
        
        if compressed:
            return vk.to_string("compressed")
        else:
            return b'\x04' + vk.to_string()
    
    def public_key_to_address(self, public_key: bytes, address_type: str = "p2pkh") -> str:
        """Convert public key to Bitcoin address"""
        if address_type == "p2pkh":
            # Legacy address (1...)
            hash160 = hashlib.new('ripemd160', hashlib.sha256(public_key).digest()).digest()
            versioned_hash = b'\x00' + hash160
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            return base58.b58encode(versioned_hash + checksum).decode()
        
        elif address_type == "p2sh":
            # Script hash address (3...)
            hash160 = hashlib.new('ripemd160', hashlib.sha256(public_key).digest()).digest()
            versioned_hash = b'\x05' + hash160
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            return base58.b58encode(versioned_hash + checksum).decode()
        
        elif address_type == "bech32":
            # Bech32 address (bc1...)
            hash160 = hashlib.new('ripemd160', hashlib.sha256(public_key).digest()).digest()
            return self.encode_bech32("bc", 0, hash160)
        
        else:
            raise ValueError(f"Unsupported address type: {address_type}")
    
    def encode_bech32(self, hrp: str, witver: int, witprog: bytes) -> str:
        """Encode Bech32 address"""
        # Simplified Bech32 encoding
        CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        
        def bech32_polymod(values):
            GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
            chk = 1
            for value in values:
                top = chk >> 25
                chk = (chk & 0x1ffffff) << 5 ^ value
                for i in range(5):
                    chk ^= GEN[i] if ((top >> i) & 1) else 0
            return chk
        
        def convertbits(data, frombits, tobits, pad=True):
            acc = 0
            bits = 0
            ret = []
            maxv = (1 << tobits) - 1
            max_acc = (1 << (frombits + tobits - 1)) - 1
            for value in data:
                if value < 0 or (value >> frombits):
                    return None
                acc = ((acc << frombits) | value) & max_acc
                bits += frombits
                while bits >= tobits:
                    bits -= tobits
                    ret.append((acc >> bits) & maxv)
            if pad:
                if bits:
                    ret.append((acc << (tobits - bits)) & maxv)
            elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
                return None
            return ret
        
        hrp_expanded = [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
        data = [witver] + convertbits(witprog, 8, 5)
        values = hrp_expanded + data + [0, 0, 0, 0, 0, 0]
        polymod = bech32_polymod(values) ^ 1
        checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
        
        return hrp + '1' + ''.join([CHARSET[d] for d in data + checksum])
    
    def generate_address_batch(self, batch_size: int = 1000) -> List[Tuple[str, str, str, str]]:
        """Generate a batch of Bitcoin addresses with different types"""
        results = []
        
        for _ in range(batch_size):
            # Generate random mnemonic
            mnemonic = self.generate_random_mnemonic()
            seed = self.mnemonic_to_seed(mnemonic)
            
            # Derive private key
            private_key = self.derive_address_key(seed)
            wif = self.private_key_to_wif(private_key)
            
            # Generate public key and addresses
            public_key = self.private_key_to_public_key(private_key, compressed=True)
            
            # Generate different address types
            p2pkh_addr = self.public_key_to_address(public_key, "p2pkh")
            p2sh_addr = self.public_key_to_address(public_key, "p2sh")
            bech32_addr = self.public_key_to_address(public_key, "bech32")
            
            results.append({
                'mnemonic': mnemonic,
                'private_key_wif': wif,
                'p2pkh_address': p2pkh_addr,
                'p2sh_address': p2sh_addr,
                'bech32_address': bech32_addr
            })
        
        return results
    
    def parallel_generate(self, total_addresses: int, num_threads: int = 8, batch_size: int = 1000) -> List[dict]:
        """Generate addresses using multiple threads"""
        results = []
        batches_per_thread = (total_addresses + batch_size - 1) // batch_size
        
        def worker():
            thread_results = []
            for _ in range(batches_per_thread // num_threads + (1 if batches_per_thread % num_threads > 0 else 0)):
                batch = self.generate_address_batch(min(batch_size, total_addresses - len(results)))
                thread_results.extend(batch)
                if len(results) >= total_addresses:
                    break
            return thread_results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
                if len(results) >= total_addresses:
                    break
        
        return results[:total_addresses]