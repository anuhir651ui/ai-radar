#!/usr/bin/env python3
"""Generate latest.json for the Weekly AI Trends & App Radar.

Data sources (all free, no keys required):
  - Hacker News via Algolia HN Search API
  - GitHub via Search API (rate-limited without auth)
  - Product Hunt daily listings via unofficial endpoint

Optional:
  Set GITHUB_TOKEN env var to increase GitHub rate limits.

Usage:
  python scripts/generate_report.py
"""

from __future__ import annotations

import os
import json
import math
import time
import re
import datetime as dt
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import requests

HN_API = "https://hn.algolia.com/api/v1/search"
GITHUB_SEARCH = "https://api.github.com/search/repositories"
PH_API = "https://www.producthunt.com/frontend/graphql"

SEARCH_QUERIES = ["AI agent", "LLM", "RAG", "AI coding", "voice AI",
                  "AI workflow", "MCP", "multimodal AI", "AI infrastructure"]

STOPWORDS = set(
    "a an and are as at be by do for from has have he how if in into is it its "
    "me my no not of on or our so that the their then there these they this to "
    "up us was we were what when which will with you your can just use using "
    "used get got one two via per also been about all any but each get has its "
    "may than too very more most own same some such only other over does did "
    "out now even still back well make made many much must way where being had "
    "would could should after before during between without through under them "
    "into made over take first last long great little just like know need want "
    "think work look find give tell try ask seem come leave call keep let begin "
    "show help start run turn move play feel put bring hear saw set found live "
    "talk point day today week month year time thing fact part case question "
    "number group problem point hacker news ycombinator github "
    "http https www com org net io dev".split()
)

NOISE_PATTERNS = re.compile(
    r"^(show|ask|tell|launch|beta|announce|release|built|building|open|"
    r"source|free|new|why|how|what|when|who|which|where|best|top|list|"
    r"looking|help|need|want|should|could|would|check|review|update|"
    r"anyone|someone|everybody|thing|stuff|post|thread|link|comment|"
    r"vote|point|discussion|question|answer|hn|yc|pg)$",
    re.IGNORECASE,
)

CATEGORY_DOMAINS = {
    "AI Agents": ["agent", "agents", "agentic", "multi-agent", "autonomous", "crew", "swarm"],
    "LLM Infrastructure": ["llm", "inference", "fine-tune", "finetune", "quantization", "serving", "deployment"],
    "RAG & Knowledge": ["rag", "retrieval", "knowledge", "embedding", "vector", "search", "indexing"],
    "AI Coding": ["coding", "code", "copilot", "programmer", "developer", "ide", "autocomplete", "cursor"],
    "Voice & Audio": ["voice", "speech", "tts", "stt", "audio", "transcription", "realtime"],
    "Multimodal": ["multimodal", "vision", "image", "video", "visual", "ocr"],
    "AI Safety & Eval": ["safety", "eval", "evaluation", "benchmark", "alignment", "guardrail", "red-team"],
    "Workflow & Orchestration": ["workflow", "orchestration", "pipeline", "chain", "dag", "automation"],
    "Data & Analytics": ["data", "analytics", "dashboard", "observability", "monitoring", "telemetry"],
}

B2B_MARKERS = [
    "sdk", "api", "framework", "orchestration", "observability", "eval",
    "deployment", "inference", "rag", "vector", "workflow", "agent",
    "enterprise", "platform", "infrastructure", "pipeline", "cloud",
    "saas", "b2b", "developer", "devtools", "backend", "server",
    "microservice", "kubernetes", "docker", "terraform", "security",
    "compliance", "soc2", "sso", "rbac", "audit",
]

CONSUMER_MARKERS = [
    "mobile", "ios", "android", "photo", "video", "music", "game",
    "keyboard", "browser", "camera", "creative", "social", "personal",
    "assistant", "chat", "consumer", "desktop", "app", "chrome",
    "extension", "plugin", "note", "writing", "productivity",
]


def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def week_window(end_date: Optional[dt.date] = None) -> Tuple[dt.date, dt.date]:
    end = end_date or utc_today()
    start = end - dt.timedelta(days=6)
    return start, end


def iso(d: dt.date) -> str:
    return d.isoformat()


def to_ts(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp())


def get_json(url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def hn_search(query: str, start_ts: int, hits: int = 50) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>{start_ts}",
        "hitsPerPage": hits,
    }
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
            "created_at": h.get("created_at"),
        })
    return out


def gh_search(start_date: dt.date, per_page: int = 30,
              token: Optional[str] = None) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []

    three_months_ago = start_date - dt.timedelta(days=90)
    six_months_ago = start_date - dt.timedelta(days=180)

    queries = [
        # Newer repos with recent traction
        f"topic:ai stars:10..50000 created:>={iso(three_months_ago)}",
        # LLM tools with moderate stars (emerging, not mega-repos)
        f"topic:llm stars:10..50000 pushed:>={iso(start_date)}",
        # Agent-specific repos
        f"topic:agent pushed:>={iso(start_date)} stars:>5",
        # AI tools found via keyword search
        f"AI agent stars:>50 pushed:>={iso(start_date)}",
        # RAG and infrastructure
        f"topic:rag pushed:>={iso(start_date)} stars:>5",
    ]

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    seen_ids: set = set()
    for i, q in enumerate(queries):
        try:
            sort_by = "updated" if i >= 2 else "stars"
            params = {"q": q, "sort": sort_by, "order": "desc", "per_page": per_page}
            data = get_json(GITHUB_SEARCH, params=params, headers=headers)
            for it in data.get("items", []):
                rid = it.get("id")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)

                created = it.get("created_at", "")
                age_days = _days_since(created) if created else 365

                all_items.append({
                    "source": "github",
                    "name": it.get("name"),
                    "full_name": it.get("full_name"),
                    "html_url": it.get("html_url"),
                    "description": it.get("description") or "",
                    "stargazers_count": it.get("stargazers_count", 0),
                    "forks_count": it.get("forks_count", 0),
                    "open_issues_count": it.get("open_issues_count", 0),
                    "language": it.get("language") or "",
                    "topics": it.get("topics", []),
                    "updated_at": it.get("updated_at"),
                    "created_at": created,
                    "age_days": age_days,
                    "license": (it.get("license") or {}).get("spdx_id", ""),
                })
            time.sleep(1.5)
        except Exception:
            continue

    all_items.sort(key=lambda x: _novelty_score(x), reverse=True)
    return all_items


def _days_since(iso_date: str) -> int:
    try:
        d = dt.datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return max(0, (dt.datetime.now(dt.timezone.utc) - d).days)
    except Exception:
        return 365


def _novelty_score(item: Dict[str, Any]) -> float:
    """Score that balances stars with recency — newer repos with decent stars rank higher."""
    stars = item.get("stargazers_count", 0)
    age_days = item.get("age_days", 365)

    star_score = 15 * math.log10(max(1, stars))
    recency_bonus = max(0, 30 - age_days * 0.08)  # New repos get up to 30 bonus points
    age_penalty = min(20, max(0, (age_days - 180) * 0.05))  # Old repos get penalized

    return star_score + recency_bonus - age_penalty


def fetch_hn_comments(object_id: str, limit: int = 10) -> List[str]:
    """Fetch top-level comments for an HN story to aid sentiment analysis."""
    try:
        url = f"https://hn.algolia.com/api/v1/search?tags=comment,story_{object_id}&hitsPerPage={limit}"
        data = get_json(url, timeout=15)
        comments = []
        for hit in data.get("hits", []):
            text = hit.get("comment_text") or ""
            clean = re.sub(r"<[^>]+>", " ", text).strip()
            if len(clean) > 20:
                comments.append(clean[:500])
        return comments
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+]{1,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and not NOISE_PATTERNS.match(w) and len(w) >= 3]


def extract_bigrams(text: str) -> List[str]:
    words = tokenize(text)
    bigrams = []
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        bigrams.append(bg)
    return bigrams


def classify_domain(text: str) -> str:
    blob = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in CATEGORY_DOMAINS.items():
        score = sum(1 for kw in keywords if kw in blob)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "General AI"
    return max(scores, key=scores.get)


def simple_sentiment(text: str) -> Dict[str, Any]:
    """Rule-based sentiment analysis using keyword matching."""
    blob = text.lower()
    positive_words = [
        "amazing", "awesome", "excellent", "fantastic", "great", "impressive",
        "innovative", "love", "powerful", "revolutionary", "solid", "stellar",
        "superb", "wonderful", "brilliant", "elegant", "fast", "reliable",
        "seamless", "intuitive", "game-changer", "breakthrough", "clean",
        "efficient", "scalable", "robust", "production-ready", "well-designed",
    ]
    negative_words = [
        "bad", "broken", "buggy", "complicated", "confusing", "disappointing",
        "expensive", "fail", "flawed", "horrible", "lacking", "mediocre",
        "poor", "slow", "terrible", "ugly", "unreliable", "unstable",
        "vendor-lock", "bloated", "overhyped", "immature", "limited",
        "insecure", "hallucinate", "hallucination", "leak", "privacy",
    ]

    pos_hits = [w for w in positive_words if w in blob]
    neg_hits = [w for w in negative_words if w in blob]

    total = len(pos_hits) + len(neg_hits)
    if total == 0:
        return {"net": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0,
                "drivers": {"positive": [], "negative": []}}

    pos_ratio = len(pos_hits) / total
    neg_ratio = len(neg_hits) / total
    neu_ratio = max(0, 1.0 - pos_ratio - neg_ratio)
    net = round(pos_ratio - neg_ratio, 2)

    return {
        "net": net,
        "pos": round(pos_ratio, 2),
        "neu": round(neu_ratio, 2),
        "neg": round(neg_ratio, 2),
        "drivers": {
            "positive": [w.replace("-", " ").title() for w in pos_hits[:5]],
            "negative": [w.replace("-", " ").title() for w in neg_hits[:5]],
        },
    }


# ---------------------------------------------------------------------------
# Trend building
# ---------------------------------------------------------------------------

def build_trends(hn_items: List[Dict[str, Any]],
                 gh_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()

    for it in hn_items:
        title = it.get("title", "")
        weight = 2 + min(3, (it.get("points", 0) // 50))
        for w in tokenize(title):
            unigram_counts[w] += weight
        for bg in extract_bigrams(title):
            bigram_counts[bg] += weight

    for it in gh_items:
        blob = f"{it.get('name', '')} {it.get('description', '')} {' '.join(it.get('topics', []))}"
        stars = it.get("stargazers_count", 0)
        weight = 1 + min(3, stars // 1000)
        for w in tokenize(blob):
            unigram_counts[w] += weight
        for bg in extract_bigrams(blob):
            bigram_counts[bg] += weight

    top_bigrams = bigram_counts.most_common(20)
    top_unigrams = unigram_counts.most_common(30)

    candidates: List[Tuple[str, int, bool]] = []
    for bg, c in top_bigrams:
        if c >= 3:
            candidates.append((bg, c, True))
    for w, c in top_unigrams:
        already_in_bigram = any(w in bg for bg, _, _ in candidates)
        if not already_in_bigram and c >= 4:
            candidates.append((w, c, False))

    candidates.sort(key=lambda x: x[1], reverse=True)

    trends = []
    used_words: set = set()

    for kw, count, is_bigram in candidates[:10]:
        words_in_kw = set(kw.split())
        if words_in_kw & used_words:
            continue
        used_words |= words_in_kw

        topic = kw.replace("-", " ").title()
        domain = classify_domain(kw)
        score = min(98, 55 + count * 3)

        evidence = []
        for h in sorted(hn_items, key=lambda x: x.get("points", 0), reverse=True):
            if kw in h.get("title", "").lower():
                evidence.append({"source": "hn", "title": h["title"], "url": h["url"]})
            if len(evidence) >= 2:
                break

        for g in gh_items:
            blob = f"{g.get('name', '')} {g.get('description', '')}".lower()
            if kw in blob:
                evidence.append({
                    "source": "github",
                    "title": g.get("full_name") or g.get("name"),
                    "url": g["html_url"],
                })
            if len(evidence) >= 4:
                break

        implications = _generate_implications(kw, domain, count)

        trends.append({
            "topic": topic,
            "domain": domain,
            "trend_score": int(score),
            "why_now": _generate_why_now(kw, count, len(evidence)),
            "implications": implications,
            "evidence": evidence,
        })
        if len(trends) >= 8:
            break

    return trends


def _generate_why_now(keyword: str, mention_count: int, evidence_count: int) -> str:
    phrases = [
        f"'{keyword}' appeared {mention_count}+ times across Hacker News discussions and trending GitHub repos this week.",
        f"Multiple sources ({evidence_count} pieces of evidence) reference '{keyword}', signaling growing builder and community interest.",
        f"Spike in mentions of '{keyword}' across developer communities, with active GitHub projects gaining traction.",
    ]
    idx = hash(keyword) % len(phrases)
    return phrases[idx]


def _generate_implications(keyword: str, domain: str, count: int) -> List[str]:
    base = []
    if domain == "AI Agents":
        base = [
            "Agent frameworks are maturing — evaluate build-vs-buy for agent capabilities.",
            "Consider agent-based workflows for repetitive PM tasks (research, summarization).",
            "Watch for multi-agent orchestration patterns becoming production-ready.",
        ]
    elif domain == "LLM Infrastructure":
        base = [
            "Inference costs continue to drop — revisit pricing models for AI features.",
            "Self-hosted LLM options expand, relevant for data-sensitive use cases.",
            "Fine-tuning workflows are becoming more accessible to product teams.",
        ]
    elif domain == "RAG & Knowledge":
        base = [
            "RAG pipelines are becoming table-stakes — invest in knowledge base quality.",
            "Hybrid search (semantic + keyword) is emerging as best practice.",
            "Evaluate vector DB options as the market consolidates.",
        ]
    elif domain == "AI Coding":
        base = [
            "AI-assisted development tools accelerate feature velocity — measure impact on your team.",
            "Code review and testing automation are emerging use cases.",
            "Developer experience is becoming a key differentiator in AI coding tools.",
        ]
    elif domain == "Voice & Audio":
        base = [
            "Real-time voice AI opens new interaction paradigms beyond text chat.",
            "Voice agent quality is approaching human parity in specific domains.",
            "Consider voice as an input modality for your product's AI features.",
        ]
    elif domain == "AI Safety & Eval":
        base = [
            "Evaluation frameworks are critical for production AI — invest early.",
            "Regulatory pressure is increasing — build safety into your AI roadmap.",
            "Guardrails and content filtering are becoming must-have features.",
        ]
    else:
        base = [
            f"Growing interest in {domain.lower()} suggests new product opportunities.",
            "Monitor this space for potential integration or partnership opportunities.",
            "Early movers in this category may establish defensible positions.",
        ]
    return base[:3]


# ---------------------------------------------------------------------------
# App building
# ---------------------------------------------------------------------------

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def score_from_stars(stars: int, age_days: int = 365) -> int:
    """Buzz score that rewards fast-growing new repos over established giants."""
    if age_days <= 0:
        age_days = 1

    star_component = 15 * math.log10(max(1, stars))

    if age_days <= 30:
        velocity = stars / max(1, age_days) * 30
        velocity_bonus = min(20, 5 * math.log10(max(1, velocity)))
    elif age_days <= 180:
        velocity = stars / max(1, age_days) * 30
        velocity_bonus = min(10, 3 * math.log10(max(1, velocity)))
    else:
        velocity_bonus = 0

    # Diminishing returns for very high star counts (avoids everything being 100)
    if stars > 50000:
        star_component = min(star_component, 55)
    elif stars > 10000:
        star_component = min(star_component, 50 + 5 * math.log10(stars / 10000))

    val = 30 + star_component + velocity_bonus
    return int(clamp(val, 0, 100))


def categorize_repo(desc: str, topics: List[str]) -> str:
    blob = (desc + " " + " ".join(topics or [])).lower()
    b2b = sum(1 for m in B2B_MARKERS if m in blob)
    consumer = sum(1 for m in CONSUMER_MARKERS if m in blob)
    if b2b > consumer:
        return "B2B SaaS"
    if consumer > b2b:
        return "Consumer"
    return "B2B SaaS"


def fit_scores(category: str, buzz: int, desc: str) -> Tuple[int, int]:
    blob = (desc or "").lower()
    b2b_base = 55 if category == "B2B SaaS" else 35
    con_base = 55 if category == "Consumer" else 30

    if any(x in blob for x in ["enterprise", "sso", "rbac", "audit", "soc2", "compliance"]):
        b2b_base += 12
    if any(x in blob for x in ["scalable", "production", "self-host", "on-premise"]):
        b2b_base += 8
    if any(x in blob for x in ["creator", "share", "video", "photo", "music", "personal"]):
        con_base += 10
    if any(x in blob for x in ["mobile", "app", "chrome", "extension", "desktop"]):
        con_base += 8

    b2b = int(clamp(b2b_base + (buzz - 70) * 0.15, 0, 100))
    con = int(clamp(con_base + (buzz - 70) * 0.12, 0, 100))
    return b2b, con


def decide_action(buzz: int, b2b_fit: int, con_fit: int, category: str,
                  stars: int = 0, age_days: int = 365) -> str:
    """Decide action using a composite score that rewards relevance + novelty."""
    fit = b2b_fit if category == "B2B SaaS" else con_fit

    novelty_boost = 0
    if age_days <= 30:
        novelty_boost = 10
    elif age_days <= 90:
        novelty_boost = 5
    elif age_days > 365 * 2:
        novelty_boost = -5

    composite = int(fit * 0.5 + buzz * 0.3 + novelty_boost)

    if composite >= 55 and fit >= 65:
        return "TRY"
    if composite >= 48 or (buzz >= 92 and fit >= 55):
        return "MONITOR"
    return "IGNORE"


def generate_watchouts(desc: str, stars: int, topics: List[str], license_id: str) -> List[str]:
    watchouts = []
    blob = (desc or "").lower()

    if stars < 100:
        watchouts.append("Very early-stage project (<100 stars) — may lack stability and support.")
    elif stars < 500:
        watchouts.append("Early-stage project — community and documentation may be thin.")

    if not license_id or license_id == "NOASSERTION":
        watchouts.append("No clear license — verify legal compatibility before integrating.")
    elif license_id in ("AGPL-3.0", "GPL-3.0"):
        watchouts.append(f"Copyleft license ({license_id}) — may restrict commercial usage.")

    if "alpha" in blob or "experimental" in blob or "proof of concept" in blob:
        watchouts.append("Described as experimental/alpha — not production-ready.")

    if any(x in blob for x in ["openai", "gpt-4", "claude", "anthropic"]):
        if not any(x in blob for x in ["local", "self-host", "ollama", "open-source"]):
            watchouts.append("Relies on proprietary API — consider vendor lock-in and cost scaling.")

    if "no documentation" in blob or len(desc or "") < 30:
        watchouts.append("Minimal description/documentation — evaluate quickly and move on if unclear.")

    return watchouts[:4]


def generate_try_plan(desc: str, category: str, topics: List[str]) -> List[str]:
    blob = (desc or "").lower()
    plan = ["Read the README and quickstart guide (5 min)"]

    if any(x in blob for x in ["api", "sdk", "library"]):
        plan.append("Install the package and run the hello-world example (10 min)")
        plan.append("Test with a real use case from your product domain (10 min)")
    elif any(x in blob for x in ["cli", "command", "terminal"]):
        plan.append("Install CLI and run the basic commands (10 min)")
        plan.append("Try processing a sample dataset or workflow (10 min)")
    elif any(x in blob for x in ["web", "dashboard", "ui"]):
        plan.append("Launch the demo/hosted version and explore the UI (10 min)")
        plan.append("Test with your own data or a representative sample (10 min)")
    else:
        plan.append("Clone the repo and run the minimal example (10 min)")
        plan.append("Evaluate core functionality against your requirements (10 min)")

    plan.append("Assess: integration complexity, data privacy, and pricing model (5 min)")
    return plan


def build_apps(gh_items: List[Dict[str, Any]],
               hn_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apps = []
    hn_text_by_keyword: Dict[str, List[str]] = {}

    for h in hn_items:
        title_lower = h.get("title", "").lower()
        for word in tokenize(title_lower):
            hn_text_by_keyword.setdefault(word, []).append(h.get("title", ""))

    for it in gh_items[:30]:
        stars = int(it.get("stargazers_count", 0))
        age_days = it.get("age_days", 365)
        buzz = score_from_stars(stars, age_days)
        desc = it.get("description", "") or ""
        topics = it.get("topics") or []
        license_id = it.get("license", "")
        category = categorize_repo(desc, topics)
        b2b_fit, con_fit = fit_scores(category, buzz, desc)
        action = decide_action(buzz, b2b_fit, con_fit, category, stars, age_days)
        domain = classify_domain(f"{desc} {' '.join(topics)}")

        sentiment_text = desc
        needle = (it.get("name") or "").lower()
        related_hn = []
        if needle:
            for h in hn_items:
                if needle in h.get("title", "").lower():
                    related_hn.append(h)
                    sentiment_text += " " + h.get("title", "")

        sentiment = simple_sentiment(sentiment_text)
        watchouts = generate_watchouts(desc, stars, topics, license_id)
        try_plan = generate_try_plan(desc, category, topics)

        evidence = [{"source": "github", "title": it.get("full_name") or it.get("name"), "url": it["html_url"]}]
        for h in related_hn:
            evidence.append({"source": "hn", "title": h["title"], "url": h["url"]})

        competitive_note = ""
        if stars > 10000:
            competitive_note = "Established player with strong community traction."
        elif stars > 1000:
            competitive_note = "Growing project with meaningful adoption signals."
        elif stars > 100:
            competitive_note = "Early-stage but gaining attention from developers."
        else:
            competitive_note = "Very new — high risk but potential for early-mover advantage."

        apps.append({
            "name": it.get("full_name") or it.get("name") or "Unknown",
            "category": category,
            "domain": domain,
            "language": it.get("language", ""),
            "license": license_id,
            "stars": stars,
            "forks": it.get("forks_count", 0),
            "age_days": age_days,
            "app_buzz_score": buzz,
            "b2b_fit": b2b_fit,
            "consumer_fit": con_fit,
            "action": action,
            "one_liner": desc.strip()[:200],
            "sentiment": sentiment,
            "competitive_note": competitive_note,
            "top_evidence": evidence,
            "try_plan_30min": try_plan,
            "watchouts": watchouts,
        })

    return apps


def build_shortlist(apps: List[Dict[str, Any]]) -> Dict[str, Any]:
    b2b_try = [a["name"] for a in apps if a.get("category") == "B2B SaaS" and a.get("action") == "TRY"][:5]
    con_try = [a["name"] for a in apps if a.get("category") == "Consumer" and a.get("action") == "TRY"][:5]
    monitor = [a["name"] for a in apps if a.get("action") == "MONITOR"][:8]
    ignore = [a["name"] for a in apps if a.get("action") == "IGNORE"][:5]

    top_pick = None
    try_apps = [a for a in apps if a.get("action") == "TRY"]
    if try_apps:
        top_pick = max(try_apps, key=lambda a: a.get("app_buzz_score", 0))
        top_pick = {
            "name": top_pick["name"],
            "buzz": top_pick["app_buzz_score"],
            "one_liner": top_pick["one_liner"],
            "reason": f"Highest buzz score ({top_pick['app_buzz_score']}) among TRY-rated apps this week.",
        }

    return {
        "b2b_try": b2b_try,
        "consumer_try": con_try,
        "monitor": monitor,
        "ignore": ignore,
        "top_pick": top_pick,
    }


def build_summary(trends: List[Dict[str, Any]], apps: List[Dict[str, Any]],
                  shortlist: Dict[str, Any]) -> Dict[str, Any]:
    domains = Counter(a.get("domain", "General AI") for a in apps)
    categories = Counter(a.get("category", "Unknown") for a in apps)
    actions = Counter(a.get("action", "MONITOR") for a in apps)
    avg_buzz = round(sum(a.get("app_buzz_score", 0) for a in apps) / max(1, len(apps)), 1)

    return {
        "total_apps_analyzed": len(apps),
        "total_trends_identified": len(trends),
        "avg_buzz_score": avg_buzz,
        "action_breakdown": dict(actions),
        "category_breakdown": dict(categories),
        "domain_breakdown": dict(domains.most_common(5)),
        "top_pick": shortlist.get("top_pick"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_date, end_date = week_window()
    start_ts = to_ts(start_date)

    print(f"Generating report for {iso(start_date)} → {iso(end_date)} ...")

    hn_items: List[Dict[str, Any]] = []
    for q in SEARCH_QUERIES:
        try:
            hn_items.extend(hn_search(q, start_ts, hits=50))
            time.sleep(0.2)
        except Exception as e:
            print(f"  HN query '{q}' failed: {e}")
            continue

    seen: set = set()
    uniq: List[Dict[str, Any]] = []
    for it in hn_items:
        key = (it.get("url"), it.get("title"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    hn_items = sorted(uniq, key=lambda x: x.get("points", 0), reverse=True)
    print(f"  Fetched {len(hn_items)} unique HN stories.")

    token = os.getenv("GITHUB_TOKEN")
    gh_items = gh_search(start_date, per_page=30, token=token)
    print(f"  Fetched {len(gh_items)} GitHub repos.")

    trends = build_trends(hn_items, gh_items)
    apps = build_apps(gh_items, hn_items)
    shortlist = build_shortlist(apps)
    summary = build_summary(trends, apps, shortlist)

    report = {
        "week_start": iso(start_date),
        "week_end": iso(end_date),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "sources": ["hn", "github"],
        "summary": summary,
        "top_trends": trends,
        "apps": apps,
        "shortlist": shortlist,
    }

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "latest.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"  {len(trends)} trends, {len(apps)} apps")
    print(f"  TRY: {summary['action_breakdown'].get('TRY', 0)}, "
          f"MONITOR: {summary['action_breakdown'].get('MONITOR', 0)}, "
          f"IGNORE: {summary['action_breakdown'].get('IGNORE', 0)}")
    if shortlist.get("top_pick"):
        print(f"  Top pick: {shortlist['top_pick']['name']} (buzz: {shortlist['top_pick']['buzz']})")


if __name__ == "__main__":
    main()
