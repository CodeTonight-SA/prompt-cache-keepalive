"""Wire prompt-cache-keepalive to the Anthropic API.

The keepalive core never imports an SDK — you inject a ``touch_fn``. This
example shows the Anthropic wiring. Run it as a sketch, not a script (it needs
a real client and your conversation prefix).

The key correctness point: the touch must re-send the SAME cached prefix (same
system / messages WITH the same ``cache_control`` breakpoint) plus one minimal
new token. The cached portion is then a cache READ — which refreshes the 5-min
TTL — and only the tiny suffix is new input.
"""
from __future__ import annotations

from prompt_cache_keepalive import KeepaliveConfig, PromptCacheKeepalive, TouchResult


def build_touch(client, model, system, prefix_messages):
    """Return a touch_fn that re-uses the cached prefix to refresh its TTL."""

    def touch() -> TouchResult:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1,  # we want the cache read, not the output
                system=system,  # last block carries cache_control={"type": "ephemeral"}
                messages=prefix_messages + [{"role": "user", "content": "."}],
            )
            usage = resp.usage
            return TouchResult(
                ok=True,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                output_tokens=usage.output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — fail soft; the loop stops on ok=False
            return TouchResult(ok=False, error=str(exc))

    return touch


def main() -> None:
    import anthropic  # local import so the package has zero hard deps

    client = anthropic.Anthropic()
    system = [{
        "type": "text",
        "text": "You are a long-lived assistant with a large cached system prompt...",
        "cache_control": {"type": "ephemeral"},
    }]
    prefix_messages = [
        {"role": "user", "content": "...the conversation so far..."},
        {"role": "assistant", "content": "..."},
    ]

    touch = build_touch(client, "claude-opus-4-8", system, prefix_messages)
    keepalive = PromptCacheKeepalive(touch, KeepaliveConfig(ttl_seconds=300, margin_seconds=30))

    # Replace with your real "is the user/agent likely to come back?" predicate.
    session_open = True
    stats = keepalive.run(is_active=lambda: session_open)
    print(f"touches={stats.touches_made} cache_read_tokens={stats.cache_read_tokens}")


if __name__ == "__main__":
    main()
