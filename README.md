# AI Market Research Agent

Automated weekly market research for product managers tracking the AI/LLM/agent ecosystem.

## What It Does

This agent runs weekly (via GitHub Actions) and produces an actionable research report:

- **Trend Detection** — Identifies emerging topics from Hacker News discussions and GitHub activity using keyword frequency analysis with bigram extraction
- **App Radar** — Evaluates trending repositories with buzz scoring, B2B/consumer fit analysis, and sentiment assessment
- **Action Recommendations** — Classifies each app as **TRY**, **MONITOR**, or **IGNORE** based on relevance scores
- **PM Insights** — Generates domain-specific implications, competitive notes, watchouts, and 30-minute try plans

## Data Sources

| Source | What's Collected | Auth Required |
|--------|-----------------|---------------|
| Hacker News (Algolia API) | Top stories mentioning AI, LLM, agents, RAG, etc. | No |
| GitHub Search API | Trending repos with AI-related topics | Optional (`GITHUB_TOKEN` increases rate limits) |

## Quick Start

### Run locally

```bash
pip install -r requirements.txt
python scripts/generate_report.py
```

This generates `latest.json` in the project root.

### View the dashboard

Open `index.html` in a browser. It loads `latest.json` and renders the interactive dashboard.

### Deploy on GitHub Pages

1. Push this repo to GitHub
2. Enable GitHub Pages (Settings → Pages → Source: main branch)
3. The dashboard is live at `https://<user>.github.io/<repo>/`
4. The GitHub Action runs every Monday to refresh the data

### Environment variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `GITHUB_TOKEN` | Increases GitHub API rate limits from 10 to 30 requests/min | Optional |

## Report Structure

The generated `latest.json` contains:

```
{
  "week_start": "2026-02-22",
  "week_end": "2026-02-28",
  "generated_at": "...",
  "sources": ["hn", "github"],
  "summary": {
    "total_apps_analyzed": 30,
    "total_trends_identified": 8,
    "avg_buzz_score": 82.5,
    "action_breakdown": {"TRY": 5, "MONITOR": 12, "IGNORE": 13},
    "top_pick": { "name": "...", "buzz": 95 }
  },
  "top_trends": [...],
  "apps": [...],
  "shortlist": {
    "b2b_try": [...],
    "consumer_try": [...],
    "monitor": [...]
  }
}
```

## Dashboard Features

- **Dark/light theme** toggle
- **Search** across app names, descriptions, and keywords
- **Filters** by category (B2B/Consumer), action (TRY/MONITOR/IGNORE), and minimum buzz score
- **Sort** by buzz score, PM fit, GitHub stars, or sentiment
- **Export** to Markdown for sharing in Slack, Notion, or docs
- **Score visualizations** with color-coded progress bars
- **Top pick banner** highlighting the week's most promising find

## How Scoring Works

### Buzz Score (0–100)
Based on GitHub stars using a logarithmic scale. A repo with 1,000 stars scores ~85, while 10,000+ stars approaches 100.

### B2B / Consumer Fit (0–100)
Keyword analysis of repo descriptions and topics against B2B markers (SDK, API, enterprise, compliance) and consumer markers (mobile, app, creative, personal).

### Action Decision
- **TRY**: High fit score (≥70) or high buzz (≥80) with decent fit (≥55)
- **MONITOR**: Moderate fit (≥50) or high buzz
- **IGNORE**: Below thresholds

### Sentiment
Rule-based keyword analysis scoring positive vs. negative language in descriptions and related HN titles.

## Project Structure

```
├── index.html                     # Dashboard (single-page app)
├── latest.json                    # Generated weekly report data
├── requirements.txt               # Python dependencies
├── scripts/
│   └── generate_report.py         # Report generation agent
├── .github/
│   └── workflows/
│       └── weekly.yml             # GitHub Actions weekly schedule
└── README.md
```
