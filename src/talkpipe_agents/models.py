"""Source/model resolution for Strands agents, the talkpipe way.

``build_agent_model`` accepts the same ``source``/``model`` strings the
talkpipe LLM segments take, applies the same config fallbacks through
talkpipe's own ``resolve_source_and_model`` (``default_model_source`` /
``default_model_name``, i.e. the ``TALKPIPE_DEFAULT_MODEL_*`` environment
variables or ``~/.talkpipe.toml``), and decodes the source through a registry that
parallels talkpipe's ``registerPromptAdapter`` — but yields a Strands
``Model`` object, which is what a ``strands.Agent`` needs (talkpipe's
prompt adapters are text-in/text-out and cannot stream tool calls).

``ollama``'s dependencies are part of the base install. The OpenAI
sources are named for the API they call: ``openai_responses`` (the
Responses API, which OpenAI's reasoning models need for tools) with
``openai`` as an alias for it, and ``openai_chat`` (Chat Completions,
which servers imitating OpenAI's API implement); all three come with the
``openai`` extra. They, ``anthropic`` and ``litellm`` import lazily and
raise an error naming this package's pip extra to install
(``talkpipe-agents[<extra>]``, which pulls in the matching
``strands-agents`` extra). Credentials are read by each provider's
SDK from its usual environment variables (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, …); the Ollama server from talkpipe's
``TALKPIPE_OLLAMA_SERVER_URL``. talkpipe's ``eliza`` prompt source has
no agent counterpart, as it cannot call tools, and the unknown-source
error says so. New sources register with
``register_agent_model(name, factory)`` where ``factory(model, *,
temperature=None, **kwargs)`` returns a Strands ``Model``; a factory is
handed ``params=`` (provider request parameters, a dict) only when the
caller set some.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Callable
from functools import cache
from importlib import import_module
from typing import Any

from strands.models import Model
from talkpipe.llm.config import resolve_source_and_model
from talkpipe.util.config import get_config
from talkpipe.util.constants import OLLAMA_SERVER_URL

ModelFactory = Callable[..., Model]

_factories: dict[str, ModelFactory] = {}

# talkpipe prompt sources that deliberately have no agent counterpart: an
# application whose configured default is one of these gets told why rather
# than just that the source is unknown.
_NON_AGENT_SOURCES = {
    "eliza": (
        "talkpipe's 'eliza' responder answers from a script and cannot call "
        "tools, so it is a llmPrompt source only"
    ),
}


def register_agent_model(name: str, factory: ModelFactory) -> None:
    """Register *factory* as agent model source *name* (the string
    segments receive as ``source=``)."""
    _factories[name] = factory


def agent_model_sources() -> list[str]:
    """The source names ``build_agent_model`` accepts (``ollama``,
    ``openai``, ``openai_responses``, ``openai_chat``, ``anthropic``,
    ``litellm`` plus
    whatever the application registered), sorted."""
    return sorted(_factories)


def build_agent_model(
    source: str | None = None,
    model: str | None = None,
    *,
    temperature: float | None = None,
    params: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> Model:
    """A Strands ``Model`` for talkpipe-style *source*/*model* strings.

    Both fall back to talkpipe's configured defaults, exactly like
    ``llmPrompt`` (the same ``resolve_source_and_model`` call, so a missing
    value raises talkpipe's own error). *params* are provider request
    parameters (``{"reasoning_effort": "none"}``, ``{"reasoning":
    {"effort": "low"}}``, Ollama ``options`` …), a dict or its JSON text
    (chatterlang has no dict literal); the provider's API defines the
    keys. Extra kwargs pass through to the source's factory.
    """
    params = model_params(params)
    if params is not None:
        kwargs["params"] = params
    # talkpipe loads its configuration once per process (get_config caches),
    # so an environment variable set after first use is not seen — a trap
    # for tests that monkeypatch the environment; docs/testing.md says so.
    try:
        source, model = resolve_source_and_model(source, model)
    except ValueError as exc:
        raise ValueError(
            f"{exc} (pass source= and model= to the agent, or set "
            "TALKPIPE_DEFAULT_MODEL_SOURCE / TALKPIPE_DEFAULT_MODEL_NAME — "
            "default_model_source / default_model_name in ~/.talkpipe.toml — "
            "before the process starts)"
        ) from exc
    factory = _factories.get(source)
    if factory is None:
        why = _NON_AGENT_SOURCES.get(source)
        hint = (
            f"; {why} — pass source= (e.g. 'ollama') to the agent, or use the "
            "'scripted' source from talkpipe_agents.testing in tests"
            if why
            else ""
        )
        raise ValueError(
            f"unknown agent model source {source!r}; registered sources: "
            f"{', '.join(agent_model_sources())}{hint}"
        )
    return factory(model, temperature=temperature, **kwargs)


def model_params(value: dict[str, Any] | str | None) -> dict[str, Any] | None:
    """Provider request parameters from a dict or its JSON text (how a
    chatterlang script spells one); ``None`` or ``""`` means none."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"model_params is not valid JSON ({exc}): {value!r}; write a "
                'JSON object, e.g. \'{"reasoning_effort": "none"}\''
            ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"model_params must be a JSON object (a dict), not {type(value).__name__}"
        )
    return value


def _import_provider(module: str, cls: str, extra: str) -> Callable[..., Model]:
    try:
        mod = import_module(f"strands.models.{module}")
    except ImportError as exc:
        raise ImportError(
            f"agent model source needs the {extra!r} extra: "
            f"pip install 'talkpipe-agents[{extra}]' "
            f"(from a checkout: pip install '.[{extra}]')"
        ) from exc
    provider_cls: Callable[..., Model] = getattr(mod, cls)
    return provider_cls


_OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def _ollama_error_hint(exc: BaseException, host: str | None, model: str) -> str | None:
    """What a newcomer needs to know about an Ollama client error that
    its own message leaves out: a refused connection (``All connection
    attempts failed``) says nothing about *where* the client looked, and
    a 404 for a model says nothing about *which* server lacks it. Returns
    ``None`` for any other error."""
    where = host or _OLLAMA_DEFAULT_HOST
    if type(exc).__name__ == "ConnectError":  # httpx, without importing it here
        return (
            f"nothing is listening at {where}: start Ollama there, or set "
            "TALKPIPE_OLLAMA_SERVER_URL to the server to use"
        )
    if getattr(exc, "status_code", None) == 404 and "not found" in str(exc):
        return (
            f"on the Ollama server at {where}; pull it there with `ollama pull {model}`"
        )
    if getattr(exc, "status_code", None) == 400 and "does not support tools" in str(
        exc
    ):
        return (
            "the agent needs a model that can call tools; pull one on "
            f"{where} (e.g. `ollama pull llama3.2`) and pass it as model="
        )
    return None


def _ollama_hint(model: Any, exc: BaseException) -> str | None:
    return _ollama_error_hint(exc, model.host, model.get_config()["model_id"])


_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


def _openai_error_hint(exc: BaseException, base_url: str | None) -> str | None:
    """What the OpenAI client's ``Connection error.`` leaves out: the
    endpoint it tried (``client_args``' ``base_url``, else
    ``OPENAI_BASE_URL``, else OpenAI itself) and the setting that moves
    it. ``None`` for any other error."""
    if type(exc).__name__ not in ("APIConnectionError", "APITimeoutError"):
        return None
    where = base_url or os.environ.get("OPENAI_BASE_URL") or _OPENAI_DEFAULT_URL
    return (
        f"could not reach {where}: start the server there, or set "
        "OPENAI_BASE_URL to the server to use"
    )


def _openai_hint(model: Any, exc: BaseException) -> str | None:
    return _openai_error_hint(
        exc, (getattr(model, "client_args", None) or {}).get("base_url")
    )


@cache
def _hinting(
    model_cls: Any, hint_of: Callable[[Any, BaseException], str | None]
) -> Callable[..., Model]:
    """*model_cls* with *hint_of*'s hint appended to the message of an
    error raised while streaming (``None`` leaves it alone); the exception
    type is unchanged. Built once per class, lazily, because the Strands
    provider modules are only imported when a model is built."""

    class HintingModel(model_cls):  # the Strands class, imported at run time
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            try:
                async for event in super().stream(*args, **kwargs):
                    yield event
            except Exception as exc:
                hint = hint_of(self, exc)
                if hint:
                    exc.args = (f"{exc} ({hint})", *exc.args[1:])
                    # ollama.ResponseError builds str() from .error, not args
                    if hasattr(exc, "error"):
                        exc.error = f"{exc.error} ({hint})"
                raise

    HintingModel.__name__ = HintingModel.__qualname__ = model_cls.__name__
    return HintingModel


def _ollama_model(
    model: str, *, temperature: float | None = None, **kwargs: Any
) -> Model:
    """Host comes from talkpipe's OLLAMA_SERVER_URL config (i.e.
    TALKPIPE_OLLAMA_SERVER_URL), like talkpipe's own Ollama adapter;
    unset means the ollama client's localhost default."""
    ollama_cls = _import_provider("ollama", "OllamaModel", "ollama")
    host = get_config().get(OLLAMA_SERVER_URL) or None
    params = kwargs.pop("params", None)
    if params:  # Ollama's request parameters are its "options" (num_ctx, top_k, …)
        kwargs["options"] = {**(kwargs.get("options") or {}), **params}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return _hinting(ollama_cls, _ollama_hint)(host, model_id=model, **kwargs)


def _params_with_temperature(
    temperature: float | None, kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    params = dict(kwargs.pop("params", None) or {})
    if temperature is not None:
        params.setdefault("temperature", temperature)
    return params or None


def _openai_chat_model(
    model: str, *, temperature: float | None = None, **kwargs: Any
) -> Model:
    """Chat Completions (``/v1/chat/completions``), the API that servers
    imitating OpenAI's (vLLM, LM Studio, gateways, …) implement."""
    openai_cls = _import_provider("openai", "OpenAIModel", "openai")
    params = _params_with_temperature(temperature, kwargs)
    return _hinting(openai_cls, _openai_hint)(model_id=model, params=params, **kwargs)


def _openai_responses_model(
    model: str, *, temperature: float | None = None, **kwargs: Any
) -> Model:
    """OpenAI's Responses API (``/v1/responses``), which its reasoning
    models require for function tools."""
    responses_cls = _import_provider(
        "openai_responses", "OpenAIResponsesModel", "openai"
    )
    params = _params_with_temperature(temperature, kwargs)
    if params is not None:  # it spreads config.get("params", {}): None breaks it
        kwargs["params"] = params
    return _hinting(responses_cls, _openai_hint)(model_id=model, **kwargs)


def _anthropic_model(
    model: str, *, temperature: float | None = None, **kwargs: Any
) -> Model:
    anthropic_cls = _import_provider("anthropic", "AnthropicModel", "anthropic")
    params = _params_with_temperature(temperature, kwargs)
    kwargs.setdefault("max_tokens", 4096)  # required by the provider
    return anthropic_cls(model_id=model, params=params, **kwargs)


def _litellm_model(
    model: str, *, temperature: float | None = None, **kwargs: Any
) -> Model:
    """Catch-all for OpenAI-compatible endpoints without a dedicated
    provider (litellm's "provider/model" naming)."""
    litellm_cls = _import_provider("litellm", "LiteLLMModel", "litellm")
    params = _params_with_temperature(temperature, kwargs)
    return litellm_cls(model_id=model, params=params, **kwargs)


register_agent_model("ollama", _ollama_model)
register_agent_model("openai_responses", _openai_responses_model)
register_agent_model("openai", _openai_responses_model)  # what OpenAI's models need
register_agent_model("openai_chat", _openai_chat_model)
register_agent_model("anthropic", _anthropic_model)
register_agent_model("litellm", _litellm_model)
