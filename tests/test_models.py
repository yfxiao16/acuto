import io

import pytest

from acuto.models import MissingCredentials, OpenAICompatModel, resolve


@pytest.mark.parametrize("factory", [OpenAICompatModel, resolve])
def test_custom_openai_endpoint_only_sends_explicit_credentials(monkeypatch, factory):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-test-key")
    requests = []

    def capture(request, *, timeout):
        requests.append(request)
        return io.BytesIO(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", capture)
    model_name = "openai:demo" if factory is resolve else "demo"
    endpoint = "http://localhost:8000/v1"

    factory(model_name, base_url=endpoint)._post({})
    factory(model_name, base_url=endpoint, api_key_env="CUSTOM_API_KEY")._post({})
    factory(
        model_name, base_url=endpoint, api_key_env="CUSTOM_API_KEY", api_key="explicit-test-key"
    )._post({})
    factory(model_name, base_url="https://api.openai.com/v1/")._post({})

    assert [request.get_header("Authorization") for request in requests] == [
        None,
        "Bearer custom-test-key",
        "Bearer explicit-test-key",
        "Bearer openai-test-key",
    ]


def test_resolve_openai_can_use_custom_endpoint_without_a_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "temporary-test-key")
    monkeypatch.delenv("CUSTOM_API_KEY")
    endpoint = "http://localhost:8000/v1"

    assert resolve("openai:demo", base_url=endpoint).api_key == ""
    with pytest.raises(MissingCredentials, match="OPENAI_API_KEY"):
        resolve("openai:demo")
    with pytest.raises(MissingCredentials, match="CUSTOM_API_KEY"):
        resolve("openai:demo", base_url=endpoint, api_key_env="CUSTOM_API_KEY")

    (tmp_path / ".env").write_text("CUSTOM_API_KEY=dotenv-test-key\n")
    assert resolve("openai:demo", base_url=endpoint, api_key_env="CUSTOM_API_KEY").api_key == (
        "dotenv-test-key"
    )


def test_openai_sanitizer_drops_undefined_required():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "ghost"],
        "additionalProperties": False,
    }
    out = OpenAICompatModel._sanitize_schema(schema)
    assert out["required"] == ["a"]
    assert schema["required"] == ["a", "ghost"]  # input untouched
    nested = {"type": "object", "properties": {"o": schema}, "required": ["o"]}
    assert OpenAICompatModel._sanitize_schema(nested)["properties"]["o"]["required"] == ["a"]


def test_bedrock_never_emits_empty_text_blocks():
    from acuto.models.bedrock import BedrockModel

    msgs = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "", "tool_calls": []},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "name": "f", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": ""},
    ]
    out = BedrockModel._to_messages(msgs)
    for m in out:
        for block in m["content"]:
            if "text" in block:
                assert block["text"].strip(), block
            if "toolResult" in block:
                assert all(c["text"].strip() for c in block["toolResult"]["content"])


def test_bedrock_sanitizes_tool_names_in_history():
    from acuto.models.bedrock import BedrockModel

    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call 1", "name": "functions.get_order()", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "call 1", "content": "x"},
    ]
    out = BedrockModel._to_messages(msgs)
    tu = out[0]["content"][0]["toolUse"]
    import re

    assert re.fullmatch(r"[a-zA-Z0-9_-]+", tu["name"]) and re.fullmatch(
        r"[a-zA-Z0-9_-]+", tu["toolUseId"]
    )
    assert out[1]["content"][0]["toolResult"]["toolUseId"] == tu["toolUseId"]
