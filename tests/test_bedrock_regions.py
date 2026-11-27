"""BedrockModel moves to another region when one region's daily quota is spent."""

from __future__ import annotations

import io
import json
import urllib.error

from acuto.models.bedrock import BedrockModel


def _reply():
    return io.BytesIO(
        json.dumps(
            {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {},
                "stopReason": "end_turn",
            }
        ).encode()
    )


class _Resp:
    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        return self.buf

    def __exit__(self, *a):
        return False


def test_a_spent_daily_quota_moves_the_call_to_the_next_region(monkeypatch):
    hosts = []

    def urlopen(req, timeout=None):
        hosts.append(req.full_url.split("/")[2])
        if len(hosts) == 1:
            body = io.BytesIO(
                b'{"message":"Too many tokens per day, please wait before trying again."}'
            )
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, body)
        return _Resp(_reply())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    m = BedrockModel(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region="us-west-2",
        access_key="k",
        secret_key="s",
        fallback_regions=["us-east-1", "us-east-2"],
    )
    turn = m.generate([{"role": "user", "content": "hi"}], tools=[])
    assert turn.text == "ok"
    assert hosts == [
        "bedrock-runtime.us-west-2.amazonaws.com",
        "bedrock-runtime.us-east-1.amazonaws.com",
    ]
    assert m.region == "us-east-1"  # it stays there for the next calls


def test_a_rate_limit_moves_on_too(monkeypatch):
    hosts = []

    def urlopen(req, timeout=None):
        hosts.append(req.full_url.split("/")[2])
        if len(hosts) == 1:
            body = io.BytesIO(b'{"message":"Too many requests, please wait before trying again."}')
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, body)
        return _Resp(_reply())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    m = BedrockModel(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region="us-east-1",
        access_key="k",
        secret_key="s",
        fallback_regions=["us-east-2"],
    )
    assert m.generate([{"role": "user", "content": "hi"}], tools=[]).text == "ok"
    assert hosts[-1] == "bedrock-runtime.us-east-2.amazonaws.com"
