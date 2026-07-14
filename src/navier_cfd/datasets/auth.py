from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


@dataclass(frozen=True)
class HuggingFaceAuth:
    """Resolved Hugging Face authentication without exposing the credential."""

    token: str | None = field(default=None, repr=False)
    source: str = "anonymous"

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "source": self.source,
            "token": "<redacted>" if self.token else None,
        }


def resolve_hf_auth(token: str | bool | None = None) -> HuggingFaceAuth:
    """Resolve credentials as explicit token > HF_TOKEN > stored login > anonymous.

    ``token=False`` explicitly disables authentication. ``token=True`` requests the
    credential saved by ``hf auth login``. Returned credentials must never be written
    to manifests, logs, checkpoints, or configuration files.
    """

    if token is False:
        return HuggingFaceAuth(None, "explicit_anonymous")
    if isinstance(token, str) and token.strip():
        return HuggingFaceAuth(token.strip(), "explicit")

    environment = os.getenv("HF_TOKEN", "").strip()
    if environment:
        return HuggingFaceAuth(environment, "environment")

    try:
        from huggingface_hub import get_token
    except ImportError:
        return HuggingFaceAuth(None, "anonymous")

    stored = get_token()
    if stored:
        return HuggingFaceAuth(stored, "stored_login")
    return HuggingFaceAuth(None, "anonymous")


__all__ = ["HuggingFaceAuth", "resolve_hf_auth"]
