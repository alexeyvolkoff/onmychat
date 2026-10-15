#!/usr/bin/env bash
# ==============================================================================
# Colab SSH Proxy Wrapper with Auto-Sync
# Автоматически находит активную виртуальную машину Colab и подключает SSH
# ==============================================================================

# Привязываем первую активную виртуалку аккаунта к имени "colab"
/home/alexey/.local/share/uv/tools/google-colab-cli/bin/python -c '
from colab_cli.common import state
from colab_cli.state import SessionState
try:
    sessions, assignments = state.sync_sessions()
    if assignments:
        a = assignments[0]
        s = SessionState(
            name="colab",
            token=a.runtime_proxy_info.token,
            url=a.runtime_proxy_info.url,
            endpoint=a.endpoint,
            token_expires_at=a.runtime_proxy_info.expires_at(),
            variant=a.variant.name,
            accelerator=a.accelerator.value,
            machine_shape=a.machine_shape.name
        )
        state.store.add(s)
except Exception:
    pass
' 2>/dev/null || true

exec /home/alexey/.local/bin/colab ssh --proxy-mode "$@"
