#!/usr/bin/env bash
# ==============================================================================
# Colab Session Sync Helper
# Связывает активную виртуальную машину Colab из браузера с локальным CLI и SSH
# ==============================================================================
/home/alexey/.local/share/uv/tools/google-colab-cli/bin/python -c '
from colab_cli.common import state
from colab_cli.state import SessionState

sessions, assignments = state.sync_sessions()
if not assignments:
    print("❌ Нет активных запущенных сессий в Google Colab.")
    print("Запустите ноутбук в браузере или выполните: colab new --gpu T4 -s colab")
else:
    for a in assignments:
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
        print(f"✅ Активная машина Colab найдена и привязана:")
        print(f"   ID:       {a.endpoint}")
        print(f"   Hardware: {a.accelerator.value} ({a.variant.name})")
        print(f"   Shape:    {a.machine_shape.name}")
        print(f"\n🚀 Теперь можно подключаться: ssh colab")
'
