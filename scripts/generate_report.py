#!/usr/bin/env python3
"""Generate latest.json for the Weekly AI Trends & App Radar static site.

Free sources (no keys required):
- Hacker News via Algolia HN Search API (public, unofficial but widely used)
- GitHub via GitHub Search API (public; rate-limited without auth)

Usage:
  python scripts/generate_report.py

Optional:
  Set GITHUB_TOKEN env var to increase GitHub rate limits.
"""

from __future__ import annotations

import os
import json
import math
import time
import re
import datetime as dt
from typing import Any, Dict, List, Tuple

import requests

HN_API = "https://hn.algolia.com/api/v1/search"
GITHUB_SEARCH = "https://api.github.com/search/repositories"

STOPWORDS = set("""a an and are as at be by for from has have if in into is it its of on or that the their then there these this to was were will with you your
ai llm gpt openai anthropic model models agent agents tool tools app apps saas launch launched new beta now""".split())

def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()

def week_window(end_date: dt.date | None = None) -> Tuple[dt.date, dt.date]:
    end = end_date or utc_today()
    start = end - dt.timedelta(days=6)
    return start, end

def iso(d: dt.date) -> str:
    return d.isoformat()

def to_ts(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp())

def get_json(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def hn_search(query: str, start_ts: int, hits: int = 25) -> List[Dict[str, Any]]:
    params = {"query": query, "tags": "story", "numericFilters": f"created_at_i>{start_ts}", "hitsPerPage": hits}
    data = get_json(HN_API, params=params)
    out = []
    for h in data.get("hits", []):
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        out.append({
            "source": "hn",
            "title": h.get("title") or "",
            "url": url,
            "points": h.get("points", 0),
            "num_comments": h.get("num_comments", 0),
            "created_at": h.get("created_at")
        })
    return out

def gh_search(start_date: dt.date, per_page: int = 30, token: str | None = None) -> List[Dict[str, Any]]:
    q = f"(topic:ai OR topic:llm OR topic:agent OR llm OR agent OR rag) pushed:>={iso(start_date)}"
    params = {"q": q, "sort": "stars", "order": "desc", "per_page": per_page}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = get_json(GITHUB_SEARCH, params=params, headers=headers)
    items = data.get("items", [])
    out = []
    for it in items:
        out.append({
            "source": "github",
            "name": it.get("name"),
            "full_name": it.get("full_name"),
            "html_url": it.get("html_url"),
            "description": it.get("description") or "",
            "stargazers_count": it.get("stargazers_count", 0),
            "forks_count": it.get("forks_count", 0),
            "open_issues_count": it.get("open_issues_count", 0),
            "topics": it.get("topics", []),
            "updated_at": it.get("updated_at")
        })
    return out

def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+]{1,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]

def build_trends(hn_items: List[Dict[str, Any]], gh_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for it in hn_items:
        for w in tokenize(it.get("title","")):
            counts[w] = counts.get(w, 0) + 2
    for it in gh_items:
        for w in tokenize((it.get("name","") + " " + it.get("description",""))):
            counts[w] = counts.get(w, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]

    trends = []
    used = set()
    for kw, c in top:
        if kw in used:
            continue
        used.add(kw)
        topic = kw.replace("-", " ").title()
        score = min(95, 60 + c * 3)
        ev = []
        for h in hn_items:
            if kw in h.get("title","").lower():
                ev.append({"source":"hn","title":h["title"],"url":h["url"]})
            if len(ev) >= 2: break
        if len(ev) < 2:
            for g in gh_items:
                blob = (g.get("name","") + " " + g.get("description","")).lower()
                if kw in blob:
                    ev.append({"source":"github","title":g.get("full_name") or g.get("name"),"url":g["html_url"]})
                if len(ev) >= 2: break

        trends.append({
            "topic": f"{topic} (buzz keyword)",
            "trend_score": int(score),
            "why_now": f"Keyword '{kw}' appears frequently across Hacker News and recently updated GitHub projects this week.",
            "implications": [],
            "evidence": ev
        })
        if len(trends) >= 5:
            break
    return trends

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def score_from_stars(stars: int) -> int:
    # 0->40 baseline, 1k->~85, 10k->~100 (diminishing returns)
    val = 40 + 15 * math.log10(max(1, stars))
    return int(clamp(val, 0, 100))

def categorize_repo(desc: str, topics: List[str]) -> str:
    blob = (desc + " " + " ".join(topics or [])).lower()
    b2b_markers = ["sdk","api","framework","orchestration","observability","eval","deployment","inference","rag","vector","workflow","agent"]
    consumer_markers = ["mobile","ios","android","photo","video","music","game","keyboard","browser","camera"]
    b2b = any(m in blob for m in b2b_markers)
    consumer = any(m in blob for m in consumer_markers)
    if b2b and not consumer: return "B2B SaaS"
    if consumer and not b2b: return "Consumer"
    return "B2B SaaS"

def fit_scores(category: str, buzz: int, desc: str) -> Tuple[int, int]:
    blob = (desc or "").lower()
    b2b_base = 55 if category == "B2B SaaS" else 35
    con_base = 55 if category == "Consumer" else 30
    if any(x in blob for x in ["enterprise","sso","rbac","audit","soc2"]): b2b_base += 10
    if any(x in blob for x in ["creator","share","video","photo","music"]): con_base += 10
    b2b = int(clamp(b2b_base + (buzz - 70) * 0.15, 0, 100))
    con = int(clamp(con_base + (buzz - 70) * 0.12, 0, 100))
    return b2b, con

def decide_action(buzz: int, b2b_fit: int, con_fit: int, category: str) -> str:
    if category == "B2B SaaS":
        if b2b_fit >= 70 or (buzz >= 80 and b2b_fit >= 55): return "TRY"
        if b2b_fit >= 50 or buzz >= 80: return "MONITOR"
        return "IGNORE"
    else:
        if con_fit >= 70 or (buzz >= 85 and con_fit >= 55): return "TRY"
        if con_fit >= 50 or buzz >= 85: return "MONITOR"
        return "IGNORE"

def build_apps(gh_items: List[Dict[str, Any]], hn_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apps = []
    for it in gh_items[:20]:
        stars = int(it.get("stargazers_count", 0))
        buzz = score_from_stars(stars)
        category = categorize_repo(it.get("description",""), it.get("topics") or [])
        b2b_fit, con_fit = fit_scores(category, buzz, it.get("description","") or "")
        action = decide_action(buzz, b2b_fit, con_fit, category)

        evidence = [{"source":"github","title":it.get("full_name") or it.get("name"),"url":it["html_url"]}]
        needle = (it.get("name") or "").lower()
        if needle:
            for h in hn_items:
                if needle in h.get("title","").lower():
                    evidence.append({"source":"hn","title":h["title"],"url":h["url"]})
                    break

        apps.append({
            "name": it.get("full_name") or it.get("name") or "Unknown",
            "category": category,
            "app_buzz_score": buzz,
            "b2b_fit": b2b_fit,
            "consumer_fit": con_fit,
            "action": action,
            "one_liner": (it.get("description") or "").strip()[:180],
            "sentiment": {"net": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "drivers": {"positive": [], "negative": []}},
            "top_evidence": evidence,
            "try_plan_30min": [
                "Scan README for value prop + quickstart",
                "Run minimal example / demo workflow",
                "Check integrations/exports and deployment story",
                "Note obvious failure modes & missing guardrails"
            ],
            "watchouts": []
        })
    return apps

def build_shortlist(apps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    b2b_try = [a["name"] for a in apps if a.get("category")=="B2B SaaS" and a.get("action")=="TRY"][:5]
    con_try = [a["name"] for a in apps if a.get("category")=="Consumer" and a.get("action")=="TRY"][:5]
    monitor = [a["name"] for a in apps if a.get("action")=="MONITOR"][:8]
    return {"b2b_try": b2b_try, "consumer_try": con_try, "monitor": monitor}

def main() -> None:
    start_date, end_date = week_window()
    start_ts = to_ts(start_date)

    hn_items: List[Dict[str, Any]] = []
    for q in ["AI", "agent", "LLM", "RAG", "voice"]:
        try:
            hn_items.extend(hn_search(q, start_ts, hits=25))
            time.sleep(0.15)
        except Exception:
            continue

    # de-dupe HN by url+title
    seen = set()
    uniq = []
    for it in hn_items:
        key = (it.get("url"), it.get("title"))
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(it)
    hn_items = uniq

    token = os.getenv("GITHUB_TOKEN")
    gh_items = gh_search(start_date, per_page=30, token=token)

    report = {
        "week_start": iso(start_date),
        "week_end": iso(end_date),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "sources": ["hn", "github"],
        "top_trends": build_trends(hn_items, gh_items),
        "apps": build_apps(gh_items, hn_items),
        "shortlist": None
    }
    report["shortlist"] = build_shortlist(report["apps"])

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "latest.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path} with {len(report['top_trends'])} trends and {len(report['apps'])} apps.")

if __name__ == "__main__":
    main()
