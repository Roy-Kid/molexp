"""Public-API regression for bundle-chat-01-class."""

from __future__ import annotations

import molexp.harness as harness
from molexp.harness import Chat


def main() -> None:
    assert "Chat" in harness.__all__
    assert "ChatMode" not in harness.__all__
    assert Chat().name == "chat"
    print("bundle-chat-01-class: ok")


if __name__ == "__main__":
    main()
