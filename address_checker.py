"""
Efficient Bitcoin address checker with parallel processing
"""

import threading
import time
from typing import Set, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import os

class AddressChecker:
    def __init__(self, addresses_file: str = "addresses.txt"):
        self.addresses_file = addresses_file
        self.target_addresses: Set[str] = set()
        self.found_addresses: List[Dict] = []
        self.lock = threading.Lock()
        self.load_target_addresses()
    
    def load_target_addresses(self):
        """Load target addresses from file"""
        if not os.path.exists(self.addresses_file):
            print(f"Warning: {self.addresses_file} not found. Creating empty file.")
            with open(self.addresses_file, 'w') as f:
                f.write("# Add Bitcoin addresses to check, one per line\n")
            return
        
        try:
            with open(self.addresses_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.target_addresses.add(line)
            
            print(f"Loaded {len(self.target_addresses)} target addresses")
        except Exception as e:
            print(f"Error loading addresses file: {e}")
    
    def reload_addresses(self):
        """Reload addresses from file (for dynamic updates)"""
        with self.lock:
            old_count = len(self.target_addresses)
            self.target_addresses.clear()
            self.load_target_addresses()
            new_count = len(self.target_addresses)
            if new_count != old_count:
                print(f"Address list updated: {old_count} -> {new_count} addresses")
    
    def check_addresses(self, generated_addresses: List[Dict]) -> List[Dict]:
        """Check generated addresses against target list"""
        matches = []
        
        for addr_data in generated_addresses:
            # Check all address types
            addresses_to_check = [
                addr_data.get('p2pkh_address'),
                addr_data.get('p2sh_address'),
                addr_data.get('bech32_address')
            ]
            
            for addr in addresses_to_check:
                if addr and addr in self.target_addresses:
                    match_data = addr_data.copy()
                    match_data['matched_address'] = addr
                    match_data['match_type'] = self.get_address_type(addr)
                    match_data['timestamp'] = time.time()
                    matches.append(match_data)
                    
                    with self.lock:
                        self.found_addresses.append(match_data)
                    
                    print(f"🎉 MATCH FOUND! Address: {addr}")
                    print(f"   Mnemonic: {addr_data['mnemonic']}")
                    print(f"   Private Key: {addr_data['private_key_wif']}")
                    print(f"   Type: {self.get_address_type(addr)}")
                    print("-" * 80)
        
        return matches
    
    def get_address_type(self, address: str) -> str:
        """Determine Bitcoin address type"""
        if address.startswith('1'):
            return 'P2PKH (Legacy)'
        elif address.startswith('3'):
            return 'P2SH (Script Hash)'
        elif address.startswith('bc1'):
            return 'Bech32 (Segwit)'
        else:
            return 'Unknown'
    
    def parallel_check(self, address_batches: List[List[Dict]], num_threads: int = 4) -> List[Dict]:
        """Check multiple batches of addresses in parallel"""
        all_matches = []
        
        def check_batch(batch):
            return self.check_addresses(batch)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_batch, batch) for batch in address_batches]
            
            for future in futures:
                matches = future.result()
                all_matches.extend(matches)
        
        return all_matches
    
    def save_matches(self, filename: str = "found_addresses.txt"):
        """Save found matches to file"""
        if not self.found_addresses:
            return
        
        with self.lock:
            try:
                with open(filename, 'a') as f:
                    for match in self.found_addresses:
                        f.write(f"=== MATCH FOUND ===\n")
                        f.write(f"Timestamp: {time.ctime(match['timestamp'])}\n")
                        f.write(f"Matched Address: {match['matched_address']}\n")
                        f.write(f"Address Type: {match['match_type']}\n")
                        f.write(f"Mnemonic: {match['mnemonic']}\n")
                        f.write(f"Private Key (WIF): {match['private_key_wif']}\n")
                        f.write(f"P2PKH Address: {match['p2pkh_address']}\n")
                        f.write(f"P2SH Address: {match['p2sh_address']}\n")
                        f.write(f"Bech32 Address: {match['bech32_address']}\n")
                        f.write("-" * 80 + "\n")
                
                print(f"Saved {len(self.found_addresses)} matches to {filename}")
                self.found_addresses.clear()  # Clear after saving
                
            except Exception as e:
                print(f"Error saving matches: {e}")
    
    def get_statistics(self) -> Dict:
        """Get checking statistics"""
        with self.lock:
            return {
                'target_addresses_count': len(self.target_addresses),
                'found_matches_count': len(self.found_addresses),
                'addresses_file': self.addresses_file
            }
    
    def add_target_address(self, address: str):
        """Add a new target address"""
        with self.lock:
            if address not in self.target_addresses:
                self.target_addresses.add(address)
                # Append to file
                try:
                    with open(self.addresses_file, 'a') as f:
                        f.write(f"{address}\n")
                    print(f"Added target address: {address}")
                except Exception as e:
                    print(f"Error adding address to file: {e}")
    
    def remove_target_address(self, address: str):
        """Remove a target address"""
        with self.lock:
            if address in self.target_addresses:
                self.target_addresses.remove(address)
                # Rewrite file
                try:
                    with open(self.addresses_file, 'w') as f:
                        f.write("# Bitcoin addresses to check, one per line\n")
                        for addr in sorted(self.target_addresses):
                            f.write(f"{addr}\n")
                    print(f"Removed target address: {address}")
                except Exception as e:
                    print(f"Error updating addresses file: {e}")