#!/usr/bin/env python3
"""Simple application health probe for container and orchestrator checks."""

import json
import os
import sys
from urllib import request, error

url = os.getenv("HEALTHCHECK_URL", "http://127.0.0.1:5000/health")
try:
    with request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode("utf-8")
        payload = json.loads(body)
        ok = resp.status == 200 and payload.get("status") in {"ok", "degraded"}
        print(json.dumps({"ok": ok, "status": resp.status, "payload": payload}))
        sys.exit(0 if ok else 1)
except (error.HTTPError, error.URLError, TimeoutError, ValueError) as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)
