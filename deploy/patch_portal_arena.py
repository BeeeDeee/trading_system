#!/usr/bin/env python3
"""Přidá záložku „Arena“ do portal/index.html (idempotentní). patch_portal_arena.py FILE; exit 3 = už tam je."""
import re
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
if 'data-tab="arena"' in s:
    sys.exit(3)
s, n1 = re.subn(r'(\n(\s*)<button role="tab" data-tab="lead"[^\n]*\n)',
                lambda m: m.group(1) + m.group(2) + '<button role="tab" data-tab="arena" aria-selected="false">Arena</button>\n',
                s, count=1)
s, n2 = re.subn(r'(\n(\s*)<iframe id="f-lead"[^\n]*\n)',
                lambda m: m.group(1) + m.group(2) + '<iframe id="f-arena" title="Arena" hidden></iframe>\n'
                + m.group(2) + '<div class="msg" id="m-arena" hidden><div>Arena zatím na tomto serveru neběží.</div></div>\n',
                s, count=1)
s, n3 = re.subn(r'lead:"/lead/"\}', 'lead:"/lead/", arena:"/arena/"}', s, count=1)
s, n4 = re.subn(r'lead:true\}', 'lead:true, arena:true}', s, count=1)
if (n1, n2, n3, n4) != (1, 1, 1, 1):
    raise SystemExit(f"struktura portal/index.html neodpovídá ({n1},{n2},{n3},{n4})")
open(p, "w", encoding="utf-8").write(s)
