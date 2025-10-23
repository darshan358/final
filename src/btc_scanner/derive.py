from __future__ import annotations

from typing import Dict, List, Literal, Tuple

from bip_utils import (
    Bip39SeedGenerator,
    Bip44, Bip44Coins, Bip44Changes,
    Bip84, Bip84Coins,
    Bip49, Bip49Coins,
)

from .utils import to_wif

AddressType = Literal["p2pkh", "p2wpkh", "p2sh-p2wpkh"]


def _get_ctx_from_seed(addr_type: AddressType, seed_bytes: bytes):
    if addr_type == "p2pkh":
        return Bip44.FromSeed(seed_bytes, Bip44Coins.BITCOIN)
    if addr_type == "p2wpkh":
        return Bip84.FromSeed(seed_bytes, Bip84Coins.BITCOIN)
    if addr_type == "p2sh-p2wpkh":
        return Bip49.FromSeed(seed_bytes, Bip49Coins.BITCOIN)
    raise ValueError("Unsupported address type")


def derive_addresses_from_mnemonic(
    mnemonic: str,
    addr_type: AddressType = "p2wpkh",
    lookahead: int = 10,
    passphrase: str = "",
) -> List[Dict[str, str]]:
    """
    Derive addresses and corresponding WIF private keys from a BIP39 mnemonic.
    Returns a list of dicts with keys: address, wif, path.
    """
    seed = Bip39SeedGenerator(mnemonic).Generate(passphrase)
    ctx = _get_ctx_from_seed(addr_type, seed)

    results: List[Dict[str, str]] = []
    acct = ctx.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT)
    if addr_type == "p2pkh":
        purpose = 44
    elif addr_type == "p2wpkh":
        purpose = 84
    elif addr_type == "p2sh-p2wpkh":
        purpose = 49
    else:
        purpose = 44
    for i in range(lookahead):
        node = acct.AddressIndex(i)
        addr = node.PublicKey().ToAddress()
        # Raw 32-byte private key
        priv_bytes = node.PrivateKey().Raw().ToBytes()
        wif = to_wif(priv_bytes, compressed=True, testnet=False)
        path = f"m/{purpose}'/0'/0'/0/{i}"
        results.append({"address": addr, "wif": wif, "path": path})
    return results
