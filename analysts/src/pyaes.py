"""Minimal Telethon import shim.

Telethon imports `pyaes` at module import time even when `cryptg` is available and
selected for the real encryption/decryption path. The local environment's `pyaes`
package build hangs, so this shim satisfies the import contract inside the
analysts-only PYTHONPATH surface.

If Telethon ever falls back to the pure-Python AES path instead of `cryptg` or
`libssl`, this shim should be replaced with the real `pyaes` package.
"""

from __future__ import annotations


class AES:
    def __init__(self, key: bytes | bytearray | list[int]) -> None:
        self.key = bytes(key)

    def encrypt(self, data: bytes | bytearray | list[int]) -> list[int]:
        raise RuntimeError('pyaes shim invoked unexpectedly; install real pyaes or ensure cryptg is available')

    def decrypt(self, data: bytes | bytearray | list[int]) -> list[int]:
        raise RuntimeError('pyaes shim invoked unexpectedly; install real pyaes or ensure cryptg is available')
