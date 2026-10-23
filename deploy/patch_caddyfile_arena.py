#!/usr/bin/env python3
"""Přidá dashboard arena5m (/arena/) do Caddyfile (idempotentní, vzor portal/patch_caddyfile_lead.py).

  patch_caddyfile_arena.py IN OUT     zapíše upravený config do OUT, exit 0; exit 3, když už IN /arena/ má.

Přidá `redir /arena /arena/` a `handle_path /arena/* { basic_auth {...}; root * /srv/arena/site; file_server }`
vedle ostatních dashboardů. basic_auth na /arena/* je povinný (SPEC D7): převezme se přihlašovací řádek
z hlavního `basic_auth` bloku webu, nebo z proměnných ARENA_AUTH_USER a ARENA_AUTH_HASH (caddy hash-password).
Zbytek souboru zůstane byte po bytu stejný.
"""
import os
import re
import sys


def credentials(s):
    user, hsh = os.environ.get("ARENA_AUTH_USER"), os.environ.get("ARENA_AUTH_HASH")
    if user and hsh:
        return f"{user} {hsh}"
    m = re.search(r"\n\tbasic_auth \{\n\t\t([^\n]+)\n\t\}\n", s)
    if not m:
        raise SystemExit("v Caddyfile není basic_auth; nastav ARENA_AUTH_USER a ARENA_AUTH_HASH")
    return m.group(1).strip()


def patch(s):
    if "/srv/arena/site" in s:
        return None
    block = ("\t# arena5m dashboard (statický snapshot JSON, zapisuje ho proces arena každých 15 s)\n"
             "\thandle_path /arena/* {\n"
             "\t\tbasic_auth {\n"
             f"\t\t\t{credentials(s)}\n"
             "\t\t}\n"
             "\t\troot * /srv/arena/site\n"
             "\t\tfile_server\n"
             "\t}\n")
    for anchor in ("lead", "evening"):
        s2, n = re.subn(rf"(\tredir /{anchor} /{anchor}/\n)", r"\1\tredir /arena /arena/\n", s, count=1)
        if n:
            s3, n2 = re.subn(rf"(\thandle_path /{anchor}/\* \{{\n(?:\t\t[^\n]*\n)*?\t\}}\n)",
                             lambda m: m.group(1) + block, s2, count=1)
            assert n2 == 1, f"blok handle_path /{anchor}/* nenalezen"
            return s3
    raise SystemExit("nenalezen redir /lead ani /evening (nejdřív portal/install.sh)")


if __name__ == "__main__":
    out = patch(open(sys.argv[1]).read())
    if out is None:
        sys.exit(3)
    open(sys.argv[2], "w").write(out)
