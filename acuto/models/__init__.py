import os

from acuto.models.anthropic import AnthropicModel
from acuto.models.base import Model, ModelTurn, ToolCall
from acuto.models.bedrock import BedrockModel
from acuto.models.gemini import GeminiModel
from acuto.models.mock import MockModel
from acuto.models.openai_compat import _OPENAI_BASE_URL, OpenAICompatModel, _key_env_for_url

_PROVIDERS = {
    "anthropic": AnthropicModel,
    "bedrock": BedrockModel,
    "gemini": GeminiModel,
    "openai": OpenAICompatModel,
}

# Credentials each provider needs, so a missing key fails here with one
# actionable line instead of inside the HTTP layer with a 401 traceback.
_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
}


class MissingCredentials(RuntimeError):
    """Raised before any request when a provider's credentials are absent."""


def _check_credentials(provider: str, kwargs: dict) -> None:
    if kwargs.get("api_key") or kwargs.get("secret_key"):
        return
    key_env = kwargs.get("api_key_env")
    if provider == "openai":
        key_env = _key_env_for_url(kwargs.get("base_url", _OPENAI_BASE_URL), key_env)
    names = (key_env,) if key_env else _KEY_ENV.get(provider, ())
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        # Same convention as the provider SDKs: the environment wins; a
        # .env in the working directory fills in what it lacks.
        from acuto.envfile import load_dotenv

        load_dotenv()
        missing = [n for n in names if not os.environ.get(n)]
    if not missing:
        return
    raise MissingCredentials(
        f"provider {provider!r} needs {' and '.join(missing)}, which is not set. "
        f"Export it, put it in a .env file in the working directory, or pass "
        f"api_key=... to resolve(). To try Acuto without any credentials, run "
        f"examples/quickstart.py, which falls back to a scripted model."
    )


def resolve(spec: str, **kwargs) -> Model:
    """Build a model from ``"provider:model-name"``.

    Providers: ``anthropic`` (ANTHROPIC_API_KEY), ``gemini``
    (GEMINI_API_KEY), ``openai`` (OPENAI_API_KEY; pass ``base_url=`` for
    any OpenAI-compatible endpoint — DeepSeek, vLLM, SGLang, ...).
    For a custom endpoint that needs authentication, pass ``api_key=`` or
    ``api_key_env=`` explicitly.

        resolve("anthropic:claude-sonnet-5")
        resolve("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0")
        resolve("gemini:gemini-2.5-flash")
        resolve("openai:gpt-5-mini")
        resolve("openai:deepseek-chat", base_url="https://api.deepseek.com/v1",
                api_key_env="DEEPSEEK_API_KEY")
    """
    provider, _, model = spec.partition(":")
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: {sorted(_PROVIDERS)}")
    if not model:
        raise ValueError(f"model name missing in spec {spec!r} (want 'provider:model')")
    _check_credentials(provider, kwargs)
    return _PROVIDERS[provider](model, **kwargs)


__all__ = [
    "Model",
    "ModelTurn",
    "ToolCall",
    "AnthropicModel",
    "BedrockModel",
    "GeminiModel",
    "MockModel",
    "OpenAICompatModel",
    "resolve",
]
