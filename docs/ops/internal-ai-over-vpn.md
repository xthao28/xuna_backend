# Internal AI over OpenVPN (temporary text provider)

Owner decision 2026-10-17: until a provider is rented, XUNA uses the owner's
internal AI, reachable only through OpenVPN. The backend talks to it through
the generic `openai_compatible` adapter, so switching later is configuration only.

## What the adapter needs

An OpenAI-style endpoint: `POST <base>/chat/completions` with `stream: true`
(Server-Sent Events). vLLM, Ollama (`/v1`), LM Studio, LiteLLM, TGI's OpenAI
route and most gateways provide it. If the internal AI has a different API,
add one adapter file under `modules/cognition/infrastructure/` — nothing else changes.

```sh
XUNA_TEXT_PROVIDER=openai_compatible
XUNA_TEXT_BASE_URL=http://<internal-host>:<port>/v1   # private address; not committed
XUNA_TEXT_MODEL=<model-name>
XUNA_TEXT_API_KEY=<if required>                      # env/secret store only
XUNA_TEXT_TIMEOUT_S=30
```

## Development (now)

1. Connect the dev machine with the normal OpenVPN client (Tunnelblick / OpenVPN Connect).
2. Check reachability: `curl -s $XUNA_TEXT_BASE_URL/models`.
3. Run the benchmark: `.venv/bin/python scripts/text_benchmark.py --runs 2`.
4. Run the API (`uvicorn ... --no-access-log`); the phone reaches the API, the API reaches the AI.
   When the VPN drops, turns return `event: error {"code": "provider_unavailable"}` and the app uses local templates.

## Hosted backend (later, if still internal)

Run an OpenVPN client as a sidecar and share its network namespace with the API,
so only the API container routes through the tunnel:

```yaml
services:
  vpn:
    image: <pinned openvpn-client image>     # choose and pin by digest at deploy time
    cap_add: [NET_ADMIN]
    devices: ["/dev/net/tun"]
    volumes: ["./secrets/client.ovpn:/vpn/client.ovpn:ro"]
  api:
    image: xuna-backend:<tag>
    network_mode: "service:vpn"
    env_file: ./secrets/api.env
```

## Rules

- `.ovpn` files, VPN credentials, internal hostnames and API keys are secrets: never in Git, logs, app or firmware.
- Conversation text is sent only for owners with U3 consent (`conversationText`), per call, and is not stored by the backend. Confirm the internal AI's own logging/retention policy before real users use it.
- Provider output is untrusted display text (backend AGENTS §1.3).
