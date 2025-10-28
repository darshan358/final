from typing import Dict, List, Optional, Tuple

from bip_utils import (
    Bip39MnemonicGenerator, Bip39SeedGenerator, Bip39WordsNum,
    Bip84, Bip49, Bip44, Bip44Coins, Bip49Coins, Bip84Coins,
)


def generate_mnemonic(num_words: int = 12) -> str:
    words_enum = {
        12: Bip39WordsNum.WORDS_NUM_12,
        15: Bip39WordsNum.WORDS_NUM_15,
        18: Bip39WordsNum.WORDS_NUM_18,
        21: Bip39WordsNum.WORDS_NUM_21,
        24: Bip39WordsNum.WORDS_NUM_24,
    }.get(num_words)
    if words_enum is None:
        raise ValueError("num_words must be one of 12,15,18,21,24")
    return Bip39MnemonicGenerator().FromWordsNumber(words_enum)


def derive_addresses(mnemonic: str, passphrase: str = "") -> Dict[str, Tuple[str, str]]:
    seed_bytes = Bip39SeedGenerator(mnemonic).Generate(passphrase)

    # BIP44 (legacy P2PKH): m/44'/0'/0'/0/0
    bip44_ctx = Bip44.FromSeed(seed_bytes, Bip44Coins.BITCOIN)
    bip44_addr = bip44_ctx.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    p2pkh_addr = bip44_addr.PublicKey().ToAddress()
    p2pkh_priv = bip44_addr.PrivateKey().ToWif()

    # BIP49 (P2SH-P2WPKH): m/49'/0'/0'/0/0
    bip49_ctx = Bip49.FromSeed(seed_bytes, Bip49Coins.BITCOIN)
    bip49_addr = bip49_ctx.Purpose().Coin().Account(0).Change(Bip49Changes.CHAIN_EXT).AddressIndex(0)
    p2sh_p2wpkh_addr = bip49_addr.PublicKey().ToAddress()
    p2sh_p2wpkh_priv = bip49_addr.PrivateKey().ToWif()

    # BIP84 (native segwit P2WPKH bech32): m/84'/0'/0'/0/0
    bip84_ctx = Bip84.FromSeed(seed_bytes, Bip84Coins.BITCOIN)
    bip84_addr = bip84_ctx.Purpose().Coin().Account(0).Change(Bip84Changes.CHAIN_EXT).AddressIndex(0)
    p2wpkh_addr = bip84_addr.PublicKey().ToAddress()
    p2wpkh_priv = bip84_addr.PrivateKey().ToWif()

    return {
        "p2pkh": (p2pkh_addr, p2pkh_priv),
        "p2sh-p2wpkh": (p2sh_p2wpkh_addr, p2sh_p2wpkh_priv),
        "p2wpkh": (p2wpkh_addr, p2wpkh_priv),
    }
