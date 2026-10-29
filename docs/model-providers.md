# Model providers

*Part of the [talkpipe-agents](../README.md) documentation.*

`llmAgent` and `AgentChat` are not tied to one model provider. `source=`
picks the provider, `model=` the model on it, and these sources are
built in:

| `source=` | Install | Credentials and endpoint | `model=` |
|-----------|---------|--------------------------|----------|
| `ollama` | included | none; the server is `localhost:11434` unless `TALKPIPE_OLLAMA_SERVER_URL` names another (`http://my-gpu-box:11434`) | a model pulled into Ollama: `gpt-oss`, `qwen3`, `llama3.2`, … |
| `openai` (alias of `openai_responses`) | `pip install 'talkpipe-agents[openai]'` | `OPENAI_API_KEY` | `gpt-4o`, … |
| `openai_chat` | `pip install 'talkpipe-agents[openai]'` | `OPENAI_BASE_URL` names the server (vLLM, LM Studio, a gateway, …); `OPENAI_API_KEY` its key (any non-empty value if it needs none) | whatever that server serves |
| `anthropic` | `pip install 'talkpipe-agents[anthropic]'` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5`, … |
| `litellm` | `pip install 'talkpipe-agents[litellm]'` | whatever LiteLLM expects for the chosen provider — its own variables, e.g. `OLLAMA_API_BASE` for `ollama_chat/<model>` (`TALKPIPE_OLLAMA_SERVER_URL` is read by the `ollama` source only) | LiteLLM's `provider/model` naming (see its [provider list](https://docs.litellm.ai/docs/providers)) |

- **Pick a model that supports tool calling.** The agent hands the
  model its tools on every call, whichever provider serves it.
- **Keys are the provider SDKs' own environment variables** (for OpenAI
  and Anthropic, the same ones talkpipe's `llmPrompt` reads), not
  talkpipe configuration keys.
- **Defaults.** Omit `source`/`model` and they fall back to talkpipe's
  `default_model_source` / `default_model_name`
  (`TALKPIPE_DEFAULT_MODEL_SOURCE` / `TALKPIPE_DEFAULT_MODEL_NAME`, or
  `~/.talkpipe.toml`), so one setting moves every such script to another
  provider. talkpipe's
  [model and source configuration guide](https://github.com/sandialabs/talkpipe/blob/stable/docs/guides/model-and-source-configuration.md)
  covers how those layers resolve.
- **Which OpenAI source?** The OpenAI sources are named for the API
  they call. For OpenAI itself, use `openai`, an alias of
  `openai_responses`: OpenAI's Responses API (`/v1/responses`), which
  its reasoning models require for function tools. (Write
  `openai_responses` to pin the API should the alias ever move.)
  `openai_chat` uses the older Chat Completions API
  (`/v1/chat/completions`) — the one servers that imitate OpenAI's API
  (vLLM, LM Studio, llama.cpp, OpenRouter, Ollama's `/v1`, corporate
  gateways) implement, and few of them implement the Responses API
  (current Ollama serves both, so either source works against its `/v1`).
  A refused connection from either source names the endpoint it tried
  and `OPENAI_BASE_URL`.
  Pointed at OpenAI itself, `openai_chat` fails for reasoning models with
  `Function tools with reasoning_effort are not supported for <model> in
  /v1/chat/completions` unless `model_params` turns reasoning off
  (below).
- **OpenAI through `litellm`.** LiteLLM sends `openai/<model>` to Chat
  Completions too, so the same reasoning models fail there with the same
  error (as `litellm.BadRequestError: OpenAIException - …`). LiteLLM
  switches to the Responses API on its own only for model names it
  recognizes and only when the request itself sets `reasoning_effort`,
  which the agent does not. Put `responses/` after the provider prefix
  to make LiteLLM use the Responses API:

  ```
  llmAgent[source="litellm", model="openai/responses/gpt-6-astra", tools=[...]]
  ```

  For OpenAI alone, `source="openai"` does the same without the LiteLLM
  layer; `litellm` earns its place when scripts move between the many
  providers it reaches.
- **`temperature=`** is passed to every built-in source. The `anthropic`
  source asks for at most 4096 output tokens per model call, since that
  provider requires a limit. Token budgets (`max_output_tokens`,
  `max_total_tokens`) work with all of them, which all report usage.
- **`model_params=`** passes provider request parameters straight
  through to the model. The provider's API defines the keys, so they
  differ between sources. chatterlang has no dict literal, so a script
  writes them as a JSON object in single quotes; from Python, `AgentChat`
  also takes a dict:

  ```
  llmAgent[source="openai", model="...", model_params='{"reasoning": {"effort": "low"}}']
  llmAgent[source="openai_chat", model="...", model_params='{"reasoning_effort": "none"}']
  llmAgent[source="ollama", model="gpt-oss", model_params='{"num_ctx": 16384}']
  ```

  `openai`/`openai_responses`, `openai_chat`, `anthropic` and `litellm` send them as
  the Strands model's `params` (with `temperature=` added unless
  `model_params` sets its own); `ollama` sends them as Ollama's
  `options`, except `replay_thinking` (below). Malformed JSON fails when
  the script compiles.
- **Ollama thinking models keep their reasoning between turns.** A
  model that reasons before its tool calls (minimax, qwen3, deepseek-r1,
  …) often decides its next steps there. The `ollama` source stores that
  reasoning on the assistant message and sends it back to Ollama as the
  message's `thinking` on later turns. Without it, such a model forgets
  its plan every turn and can search until it runs out of turns. A
  turn's text and all its tool calls go back as one assistant message,
  and each tool result carries its `tool_name`, as Ollama's chat API
  expects. Models that do not think are unaffected. Replayed reasoning
  takes up context, so a model with a small `num_ctx` may reach it
  sooner. If a model reacts badly to the replay, turn it off with
  `model_params='{"replay_thinking": false}'`.
- **Not talkpipe's `eliza`.** talkpipe's scripted `eliza` responder is a
  `llmPrompt` source only — it cannot call tools, so it is not an agent
  source. If your configured default source is `eliza`, pass `source=`
  to `llmAgent`/`AgentChat`; otherwise the agent stops with `unknown
  agent model source 'eliza'`. For offline tests use the `scripted`
  source described in [Testing without a model server](testing.md).

## Any other provider

Any other provider is one `register_agent_model(name, factory)`
call, where `factory(model, *, temperature=None, **kwargs)` returns a
Strands `Model` and *name* is what scripts then pass as `source=`
(`agent_model_sources()` lists the names registered so far).
When the script sets `model_params`, the factory receives them as
`params=` (a dict); otherwise no `params` keyword is passed.
Strands ships more providers than the four registered here (Amazon
Bedrock, Gemini, Mistral, llama.cpp, …), and a factory is also the way
to hand a provider explicit settings instead of environment variables.
Whichever Strands model class the factory imports needs its provider
package installed — the example below reuses the OpenAI client for an
OpenAI-compatible server, so it needs `pip install
'talkpipe-agents[openai]'` (or the provider's own `strands-agents`
extra for a class with no extra here):

```python
from strands.models.openai import OpenAIModel

from talkpipe_agents import register_agent_model


def gpu_box(model, *, temperature=None, **kwargs):
    params = {"temperature": temperature} if temperature is not None else None
    return OpenAIModel(
        client_args={"base_url": "http://my-gpu-box:8000/v1", "api_key": "unused"},
        model_id=model,
        params=params,
    )


register_agent_model("gpu_box", gpu_box)  # llmAgent[source="gpu_box", model="..."]
```

Under the hood, `build_agent_model(source, model)` takes `source`/`model`
strings of the same form talkpipe's own LLM segments take, applies
the same configuration fallbacks by calling talkpipe's
`resolve_source_and_model` (`default_model_source` /
`default_model_name`, i.e. the `TALKPIPE_DEFAULT_MODEL_*`
environment variables or `~/.talkpipe.toml`), and decodes the source
through this registry of factories. Ollama works out of the box
(`TALKPIPE_OLLAMA_SERVER_URL` names the server; unset means
localhost); the OpenAI (both APIs), Anthropic, and LiteLLM factories are
registered too but import lazily behind their pip extras.
