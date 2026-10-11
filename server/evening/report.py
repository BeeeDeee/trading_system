"""Static dashboard for the server: dashboard.src.html with the stored data inlined, published atomically.

The artifact page reads its data from the artifact database; here the same page gets the same documents inline
(<script type="application/json">) so the single executable script stays byte-identical between builds and the CSP
can pin its hash. Output: index.html + data.json under <web>/releases/<ts>/, symlink <web>/current.
"""
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil

from . import cal
from .store import Store

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dashboard", "dashboard.src.html")
KEEP_RELEASES = 5

STALE_JS = """
(function(){
  try{
    var td=JSON.parse(document.getElementById("tradingDays").textContent),runs=SAMPLE.runs||[];
    var have={};runs.forEach(function(r){have[r.date]=r.status});
    var now=Date.now(),exp=null;
    td.forEach(function(d){ if(Date.parse(d+"T21:30:00Z")<=now) exp=d; });
    if(exp&&!have[exp]){
      var b=document.getElementById("sampleBanner");
      b.textContent="Chybí záznam o běhu za "+exp+" (poslední uložený: "+(runs.length?runs[runs.length-1].date:"žádný")+"). Zkontroluj systemd timer a journal.";
      b.hidden=false;b.style.background="var(--neg-bg)";b.style.color="var(--neg)";
    }
  }catch(e){}
})();
"""


def collect(st):
    prices = [{"date": p["date"], "close": p["close"]} for p in st.all("prices")]
    return {"days": st.all("days"), "trades": st.all("trades"), "runs": st.all("runs"), "scores": st.all("scores"),
            "prices": prices, "config": st.config()}


def build_html(data, today=None):
    src = open(SRC, encoding="utf-8").read()
    if "__SAMPLE__" not in src or "{...SAMPLE,sample:true}" not in src:
        raise RuntimeError("dashboard.src.html changed: placeholder or sample fallback not found")
    src = re.sub(r'<link rel="preconnect"[^>]*>\s*<link rel="stylesheet" href="https://fonts[^>]*>\s*', "", src)   # external fonts: blocked by the CSP
    src = src.replace("{...SAMPLE,sample:true}", "{...SAMPLE,sample:false}")                                         # the "sample" data IS the real data
    src = src.replace("__SAMPLE__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    # no artifact runtime here: drop the database subscription block
    i = src.index("(async()=>{\n  const db=await window.claude")
    j = src.index("})();\n})();", i)
    src = src[:i] + src[j + len("})();\n"):]
    days = []
    d = (dt.date.today() - dt.timedelta(days=10))
    while d <= dt.date.today() + dt.timedelta(days=400):
        if cal.is_trading_day(d):
            days.append(d.isoformat())
        d += dt.timedelta(days=1)
    main_close = src.rindex("</script>")
    src = src[:main_close] + STALE_JS + src[main_close:]
    src = src.replace('<script id="sampleData" type="application/json">', '<script id="tradingDays" type="application/json">' + json.dumps(days) + '</script>\n<script id="sampleData" type="application/json">', 1)
    scripts = re.findall(r"<script(?![^>]*type=\"application/json\")[^>]*>(.*?)</script>", src, re.S)
    hashes = " ".join("'sha256-" + base64.b64encode(hashlib.sha256(x.encode("utf-8")).digest()).decode() + "'" for x in scripts)
    csp = (f'<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src {hashes}; style-src \'unsafe-inline\'; '
           f'img-src \'self\' data:; base-uri \'none\'; form-action \'none\'">')
    return ('<!doctype html><html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            + csp + "</head><body>\n" + src + "\n</body></html>\n")


def publish(data_dir, web_dir):
    st = Store(data_dir)
    data = collect(st)
    if not data["days"] or not data["config"]:
        raise RuntimeError("no data to publish")
    html = build_html(data)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    rel = os.path.join(web_dir, "releases", ts)
    os.makedirs(rel)
    with open(os.path.join(rel, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(rel, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    for fn in ("index.html", "data.json"):
        os.chmod(os.path.join(rel, fn), 0o644)
    os.chmod(rel, 0o755)
    link, tmp = os.path.join(web_dir, "current"), os.path.join(web_dir, "current.new")
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.join("releases", ts), tmp)
    os.replace(tmp, link)
    rels = sorted(os.listdir(os.path.join(web_dir, "releases")))
    for old in rels[:-KEEP_RELEASES]:
        shutil.rmtree(os.path.join(web_dir, "releases", old), ignore_errors=True)
    return rel
