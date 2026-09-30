"""Tamper-evident hash chain over all run and correction records (data/chain.jsonl, append-only).

this_hash = sha256(canonical JSON of the record without this_hash), and the record contains prev_hash and the
sha256 of every file of the run. Changing any byte of any run breaks the chain from that point on.
"""
import json
import os

from . import canon

GENESIS = "0" * 64


def chain_path(repo):
    return os.path.join(repo, "data", "chain.jsonl")


def entries(repo):
    p = chain_path(repo)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def last_hash(repo):
    e = entries(repo)
    return e[-1]["this_hash"] if e else GENESIS


def files_digest(run_dir, exclude=("run.json",)):
    out = {}
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("failed-") and not d.startswith(".work"))
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), run_dir)
            if rel in exclude:
                continue
            out[rel] = canon.sha256_file(os.path.join(root, fn))
    return dict(sorted(out.items()))


def record_hash(rec):
    return canon.sha256_obj({k: v for k, v in rec.items() if k != "this_hash"})


def seal(rec, prev_hash):
    rec = dict(rec, prev_hash=prev_hash)
    rec["this_hash"] = record_hash(rec)
    return rec


def append(repo, rec_type, date, rel_path, this_hash):
    e = entries(repo)
    line = {"seq": len(e) + 1, "type": rec_type, "date": date, "path": rel_path, "this_hash": this_hash}
    p = chain_path(repo)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(canon.dumps(line) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return line


def verify(repo):
    """Structural check of the chain: record hashes, links, file digests. Returns list of problems."""
    probs = []
    prev = GENESIS
    for i, e in enumerate(entries(repo), 1):
        if e["seq"] != i:
            probs.append(f"seq {e['seq']} != {i}")
        path = os.path.join(repo, e["path"])
        rec = canon.read_json(path)
        if rec is None:
            probs.append(f"{e['path']}: chybí")
            continue
        if rec.get("prev_hash") != prev:
            probs.append(f"{e['path']}: prev_hash nesedí")
        if record_hash(rec) != rec.get("this_hash") or rec.get("this_hash") != e["this_hash"]:
            probs.append(f"{e['path']}: this_hash nesedí (záznam byl změněn)")
        if rec.get("files") is not None:
            now = files_digest(os.path.dirname(path), exclude=(os.path.basename(path),))
            if now != rec["files"]:
                diff = sorted(set(now.items()) ^ set(rec["files"].items()))
                probs.append(f"{e['path']}: soubory běhu se liší od otisků ({', '.join(sorted({d[0] for d in diff})[:5])})")
        prev = e["this_hash"]
    return probs
