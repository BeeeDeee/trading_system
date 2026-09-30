"""HTTP GET with verbatim raw capture (URL, time, status, sha256) for the audit trail."""
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request

from . import canon

UA = "crypto-paper-bot/1.0 (paper trading research; contact via repo)"


class FetchError(Exception):
    pass


class Recorder:
    """Stores every response body verbatim in raw_dir and keeps a manifest.

    Bodies of intraday requests are gzip-compressed (mtime=0, deterministic); sha256 is always of
    the uncompressed bytes exactly as received.
    """

    def __init__(self, raw_dir, clock, secrets=None):
        self.raw_dir = raw_dir
        self.clock = clock
        self.entries = []
        self.secrets = secrets or {}          # header values never written to the manifest
        self._lock = threading.Lock()         # parallel fetches share one manifest
        os.makedirs(raw_dir, exist_ok=True)

    def get(self, url, name, kind, meta=None, headers=None, timeout=20, retries=2, compress=False):
        hdrs = {"User-Agent": UA, "Accept": "application/json"}
        hdrs.update(headers or {})
        last_err = None
        for attempt in range(retries + 1):
            t0 = self.clock.now_ms()
            status, body, err = None, b"", None
            try:
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    status, body = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, body, err = e.code, e.read() or b"", f"HTTP {e.code}"
            except Exception as e:  # network error, timeout
                err = f"{type(e).__name__}: {e}"
            data = gzip.compress(body, mtime=0) if compress else body
            with self._lock:
                seq = len(self.entries) + 1
                fname = f"{seq:04d}_{name}.json" + (".gz" if compress else "")
                self.entries.append(None)
            canon.write_bytes(os.path.join(self.raw_dir, fname), data)
            self.entries[seq - 1] = ({
                "seq": seq, "file": fname, "url": url, "kind": kind, "meta": meta or {},
                "fetched_at": canon.ms_iso(t0), "status": status, "error": err,
                "sha256": canon.sha256_bytes(body), "bytes": len(body), "gzip": compress,
            })
            if err is None and status == 200:
                try:
                    return json.loads(body)
                except ValueError as e:
                    last_err = f"invalid JSON from {url}: {e}"
            else:
                last_err = f"{err or status} from {url}"
                if status in (400, 404, 451):          # permanent: do not hammer
                    break
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1) if not getattr(self.clock, "simulated", False) else 0)
        raise FetchError(last_err)

    def manifest(self):
        """Entries in a stable order (parallel fetches complete in any order): by kind, coin, URL, attempt."""
        es = [e for e in self.entries if e]
        return {"entries": sorted(es, key=lambda e: (e["kind"], str(e["meta"].get("coin", "")), e["url"], e["seq"]))}


def read_raw(raw_dir, entry):
    with open(os.path.join(raw_dir, entry["file"]), "rb") as f:
        data = f.read()
    return gzip.decompress(data) if entry.get("gzip") else data
