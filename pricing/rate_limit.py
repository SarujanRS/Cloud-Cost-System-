"""
rate_limit.py
--------------
Brute-force protection for login: an in-memory sliding-window counter, keyed
separately by username (protects one account against credential stuffing from
anywhere) and by client IP (protects against one source hammering many
usernames). No new dependency, consistent with the rest of pricing/ using
stdlib only where practical.

Scope note: state is in-memory and per-process, so it resets on a server
restart and doesn't share state across multiple worker processes. Fine for
this single-process dev server; a production deployment with multiple workers
would want this backed by a shared store (e.g. Redis) instead.
"""

import time
import threading

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 15 * 60
LOCKOUT_SECONDS = 15 * 60

MAX_ATTEMPTS_PER_IP = 20
IP_WINDOW_SECONDS = 15 * 60
IP_LOCKOUT_SECONDS = 15 * 60

_lock = threading.Lock()
_by_username = {}  # username -> list[float] failure timestamps
_by_ip = {}  # ip -> list[float] failure timestamps
_locked_until = {}  # ("user"|"ip", key) -> unix timestamp


def _prune(timestamps, window_seconds, now):
    cutoff = now - window_seconds
    return [t for t in timestamps if t > cutoff]


def _seconds_remaining(kind, key, now):
    until = _locked_until.get((kind, key))
    if until is None or until <= now:
        return 0
    return int(until - now)


def check_locked_out(username, ip):
    """Returns seconds remaining if locked out, else 0. Checked before verifying
    a password so a locked-out attacker doesn't get a free extra guess."""
    now = time.time()
    with _lock:
        user_wait = _seconds_remaining("user", (username or "").strip().lower(), now)
        ip_wait = _seconds_remaining("ip", ip, now)
        return max(user_wait, ip_wait)


def record_failure(username, ip):
    now = time.time()
    uname = (username or "").strip().lower()
    with _lock:
        attempts = _prune(_by_username.get(uname, []), WINDOW_SECONDS, now)
        attempts.append(now)
        _by_username[uname] = attempts
        if len(attempts) >= MAX_ATTEMPTS:
            _locked_until[("user", uname)] = now + LOCKOUT_SECONDS

        ip_attempts = _prune(_by_ip.get(ip, []), IP_WINDOW_SECONDS, now)
        ip_attempts.append(now)
        _by_ip[ip] = ip_attempts
        if len(ip_attempts) >= MAX_ATTEMPTS_PER_IP:
            _locked_until[("ip", ip)] = now + IP_LOCKOUT_SECONDS


def record_success(username, ip):
    uname = (username or "").strip().lower()
    with _lock:
        _by_username.pop(uname, None)
        _locked_until.pop(("user", uname), None)
        # A successful login only clears that username's counter, not the IP's -
        # one guessed/known account shouldn't wipe out evidence of a broader
        # IP-level spray against other accounts.
