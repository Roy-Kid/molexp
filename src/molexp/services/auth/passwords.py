"""Argon2id password hashing — the only password codec for molexp auth."""

from __future__ import annotations


def hash_password(password: str) -> str:
    """Return a PHC-string argon2id hash of *password*."""
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Return True when *password* matches *password_hash*."""
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    try:
        return PasswordHasher().verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
