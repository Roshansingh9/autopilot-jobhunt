import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from tinyfish import RateLimitError, TinyFish

from job_hunt.llm_utils import chat_with_llm
from job_hunt.log import get_logger
from job_hunt.notifier import send_telegram

logger = get_logger()

STATE_FILE = Path("state/seen_jobs.json")
LAST_SCAN_FILE = Path("state/last_scan.json")
JOB_HISTORY_FILE = Path("state/job_history.json")

# ── Deterministic pre-filter constants ──────────────────────────────────────

_BLOCKED_SENIORITY = frozenset({
    "senior", "principal", "lead", "manager", "director",
    "architect", "vp", "head",
    # "staff" handled separately: "Staff Engineer" = blocked, "Member of Technical Staff" = allowed
})

_BLOCKED_FUNCTIONS = frozenset({
    "marketing", "sales", "finance", "hr", "recruiter", "recruiting",
    "talent acquisition", "customer success", "legal",
    "business development", "design", "graphic", "product marketing",
    "account manager", "account executive", "partnership", "brand",
    "support specialist", "customer support",
})

_ENGINEERING_KEYWORDS = frozenset({
    # Role titles
    "software", "engineer", "developer", "programmer", "scientist",
    "sde", "swe", "mts", "technical staff",
    # Specializations
    "backend", "ai", "ml", "machine learning", "platform", "infrastructure",
    "distributed", "data", "sre", "devops", "site reliability",
    "fullstack", "full stack", "cloud", "systems", "mlops",
    "applied", "research", "analytics", "security", "embedded", "firmware",
    # Entry-level signals (catch "Graduate Software Engineer", "Associate Engineer")
    "graduate", "associate",
})

_MAX_JOBS_PER_COMPANY = 20

# ── Rule-based fallback scoring ──────────────────────────────────────────────
# Used when all LLM models fail quota. Priority: role type > level > location > stack.

# Title words that signal a core engineering role (word-level match)
_RULE_ENG_WORDS = frozenset({
    "engineer", "developer", "programmer", "scientist",
    "sde", "swe", "mts",
})
_RULE_PLATFORM_WORDS = frozenset({
    "platform", "infrastructure", "backend", "devops", "sre",
    "reliability", "systems", "distributed",
})
# Title words that signal entry / early career
_RULE_ENTRY_WORDS = frozenset({
    "i", "ii", "1", "2", "junior", "graduate", "grad",
    "associate", "entry", "new", "fresher", "trainee",
})

_RULE_SKILLS = frozenset({
    # Core CS / engineering fundamentals (high value — broad applicability)
    "distributed systems", "microservices", "backend", "api", "scalable",
    "low latency", "high availability", "data structures", "algorithms",
    # Languages (moderate value)
    "python", "java", "c++", "golang", "go", "rust", "typescript",
    # Cloud / infra
    "aws", "gcp", "azure", "kubernetes", "docker",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "kafka",
    # AI/ML
    "machine learning", "deep learning", "llm", "pytorch", "tensorflow",
    # Frameworks (LOW value — do not dominate)
    "fastapi", "django", "spring", "react", "node",
})

_RULE_LOCATION = frozenset({
    "india", "remote", "hybrid", "bangalore", "bengaluru", "mumbai",
    "delhi", "hyderabad", "pune", "chennai", "gurgaon", "noida",
    "new delhi", "gurugram",
})


def _rule_based_score(job: dict, config: dict) -> dict:
    """Deterministic fallback scorer: role type first, stack last."""
    title = (job.get("title") or "").lower()
    content = (job.get("content") or "").lower()
    location = (job.get("location") or "").lower()
    title_words = set(re.split(r"[\s\-_/,|.]+", title))
    combined = f"{title} {content[:600]}"
    loc_text = f"{location} {content[:300]}"

    score = 20  # base

    # 1. Role type (most important)
    if title_words & _RULE_ENG_WORDS:
        score += 30
    elif title_words & _RULE_PLATFORM_WORDS:
        score += 22

    # 2. Entry-level / level fit
    if title_words & _RULE_ENTRY_WORDS:
        score += 10

    # 3. Location
    if any(loc in loc_text for loc in _RULE_LOCATION):
        score += 10

    # 4. Tech stack — small bonus, capped low so it can't dominate
    skills_hit = sum(1 for s in _RULE_SKILLS if s in combined)
    score += min(skills_hit * 3, 15)

    score = min(score, 82)

    min_score = config.get("candidate", {}).get("min_score", 55)
    result = job.copy()
    result.update({
        "score": score,
        "extracted_title": job.get("title", ""),
        "stack": "",
        "location_remote": job.get("location", ""),
        "reason": f"[rule-based fallback, LLM unavailable] score={score}",
        "worth_applying": score >= min_score,
    })
    return result


JOB_URL_RE = re.compile(
    r"/(job|jobs|opening|openings|position|positions|vacancy|vacancies|role|roles|apply)"
    r"/[a-zA-Z0-9_%@.-]{4,}",
    re.IGNORECASE,
)
ATS_JOB_RE = re.compile(
    r"(greenhouse\.io/.+/jobs/\d+"
    r"|lever\.co/[^/]+/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    r"|myworkdayjobs\.com/[^?#]+"
    r"|smartrecruiters\.com/[^/]+/[A-Z0-9]+"
    r"|ashbyhq\.com/[^/]+/[a-f0-9-]{32,})",
    re.IGNORECASE,
)
ATS_LISTING_RE = re.compile(
    r"^https?://(jobs\.lever\.co|boards\.greenhouse\.io|apply\.workable\.com"
    r"|jobs\.smartrecruiters\.com)/[^/?#]+/?(\?.*)?$",
    re.IGNORECASE,
)

SEARCH_QUERY = (
    'site:{domain} '
    '("software engineer" OR "software developer" OR "SDE" OR "backend engineer" '
    'OR "platform engineer" OR "infrastructure engineer" OR "AI engineer" '
    'OR "ML engineer" OR "machine learning engineer" OR "graduate engineer" '
    'OR "associate engineer" OR "member of technical staff")'
)

_SCORE_SYSTEM = (
    "You are a job-match scoring API. "
    "Reply with ONLY a valid JSON array. "
    "No markdown, no code fences, no explanation. "
    "Start with [ and end with ]."
)

SCORE_PROMPT = """Score {job_count} jobs for this candidate.
Output ONLY a JSON array of {job_count} objects — nothing else.

CANDIDATE: {candidate_profile}
BACKGROUND: {resume_summary}

SCORING PRIORITY (apply in this order — stack is LAST):
1. ROLE TYPE — Software Engineer / SDE / Backend / Platform / Infrastructure / AI / ML / MTS = strong fit. Non-engineering = reject.
2. LEVEL FIT — Entry, junior, new-grad, SDE-I/II, Graduate, Associate, MTS-I = excellent for this 2026 CS grad. Senior/Staff/Principal/Lead/Manager = very poor (overqualified req).
3. LOCATION — India or remote-from-India = bonus. On-site abroad with no remote = penalty.
4. ENGINEERING DEPTH — interesting technical problem, backend/distributed/systems/AI = bonus.
5. TECH STACK — MINOR bonus only. Candidate learns any stack quickly. Do NOT penalize absence of a specific framework.

BOOST heavily: Software Engineer, SDE, Graduate Engineer, Associate Engineer, Backend Engineer, Platform Engineer, Infrastructure Engineer, Systems Engineer, AI/ML Engineer, MTS, Core Engineer, Software Developer.
PENALIZE heavily: Senior, Staff, Principal, Lead, Architect, Manager, Director, VP, Head of — candidate is early-career.
PENALIZE: Marketing, Sales, HR, Finance, Legal, Design, Operations, Support, Customer Success, Business Development.

JOBS:
{jobs_text}

Required schema per object:
{{"job_number":1,"score":0-100,"title":"job title","stack":"t1,t2","location_remote":"place/policy","reason":"one sentence","worth_applying":true}}

Scoring: 80-100 = SDE/SE/BE role at good company in India or remote, good level fit; 60-79 = good fit; 40-59 = partial; <40 = poor.
worth_applying=true only if score>={min_score}.
Output ONLY the JSON array. Begin with ["""

EXPORT_FIELDS = [
    "Company", "Role", "Location", "Application URL",
    "Score (%)", "Stack", "Region", "Reason", "Worth Applying", "Scan Date",
]


def _build_candidate_profile(config: dict) -> str:
    cand = config.get("candidate", {})
    name = cand.get("name", "the candidate")
    profile = cand.get("profile", "")
    seeking = cand.get("seeking", "")
    not_suitable = cand.get("not_suitable", "")

    lines = [f"- {name}"]
    if profile:
        lines.append(f"- {profile}")
    if seeking:
        lines.append(f"- Seeking: {seeking}")
    if not_suitable:
        lines.append(f"- NOT suitable: {not_suitable}")
    return "\n".join(lines)


def is_job_url(url: str) -> bool:
    return bool(JOB_URL_RE.search(url)) or bool(ATS_JOB_RE.search(url))


def is_ats_listing(url: str) -> bool:
    return bool(ATS_LISTING_RE.match(url))


# ── Pre-filter helpers ───────────────────────────────────────────────────────

def _has_blocked_seniority(title: str) -> bool:
    t = title.lower()
    words = set(re.split(r"[\s\-/,|]+", t))
    if words & _BLOCKED_SENIORITY:
        return True
    # "staff" alone means senior (e.g. "Staff Engineer"), but
    # "Member of Technical Staff" and "MTS" are entry-level at many companies.
    if "staff" in words and "member" not in words:
        return True
    return False


def _has_blocked_function(title: str) -> bool:
    t = title.lower()
    return any(fn in t for fn in _BLOCKED_FUNCTIONS)


def _has_engineering_keyword(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _ENGINEERING_KEYWORDS)


def _prefilter_by_title(jobs: list[dict]) -> tuple[list[dict], int]:
    """Filter on URL-slug title before fetching job content (fast, zero API cost)."""
    kept, discarded = [], 0
    for job in jobs:
        title = job.get("title", "")
        # Numeric-only titles come from ATS job IDs — can't determine, keep them
        if re.fullmatch(r"\d+", title.strip()):
            kept.append(job)
            continue
        if _has_blocked_seniority(title) or _has_blocked_function(title):
            discarded += 1
            continue
        kept.append(job)
    return kept, discarded


def _prefilter_by_content(jobs: list[dict]) -> tuple[list[dict], int]:
    """Filter on real title + fetched content (runs after fetch_job_details)."""
    kept, discarded = [], 0
    for job in jobs:
        title = job.get("title") or ""
        content = job.get("content", "")
        if re.fullmatch(r"\d+", title.strip()):
            kept.append(job)
            continue
        if _has_blocked_seniority(title) or _has_blocked_function(title):
            discarded += 1
            continue
        # Require at least one engineering keyword in title; fall back to content
        if title and not _has_engineering_keyword(title):
            if not _has_engineering_keyword(f"{title} {content[:300]}"):
                discarded += 1
                continue
        kept.append(job)
    return kept, discarded


def _extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from LLM output that may contain markdown or prose."""
    text = text.strip()
    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # 2. Markdown code block
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # 3. Outermost [...] span
    start, end = text.find("["), text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def load_state(state_file: Path = STATE_FILE) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"seen_urls": []}


def save_state(state: dict, state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


_FETCH_URL_DELAY = 2.5


def _fetch_with_ratelimit(tf: TinyFish, urls: list[str], **kwargs):
    for attempt in range(2):
        try:
            resp = tf.fetch.get_contents(urls, **kwargs)
            time.sleep(len(urls) * _FETCH_URL_DELAY)
            return resp
        except RateLimitError:
            logger.warning("Fetch rate-limited — waiting 65s before retry...")
            time.sleep(65)
        except Exception as e:
            logger.error(f"Fetch error for {urls[:1]}: {e}")
            time.sleep(len(urls) * _FETCH_URL_DELAY)
            return None
    return None


def _fetch_links(tf: TinyFish, urls: list[str]) -> dict[str, list[str]]:
    result = {}
    for i in range(0, len(urls), 10):
        batch = urls[i: i + 10]
        resp = _fetch_with_ratelimit(tf, batch, format="markdown", links=True)
        if resp:
            for r in resp.results:
                result[r.url] = r.links
    return result


def discover_job_urls(tf: TinyFish, company: dict, seen_urls: set) -> list[dict]:
    found_urls: set[str] = set()

    logger.debug(f"  [{company['name']}] Fetching careers page: {company['careers_url']}")
    resp = _fetch_with_ratelimit(tf, [company["careers_url"]], format="markdown", links=True)
    if resp and resp.results:
        links = resp.results[0].links
        direct = [link for link in links if is_job_url(link) and link not in seen_urls]
        ats_pages = list({link for link in links if is_ats_listing(link)})
        found_urls.update(direct)
        logger.debug(f"  [{company['name']}] Careers page: {len(direct)} direct job links, {len(ats_pages)} ATS listing pages")

        if ats_pages:
            logger.debug(f"  [{company['name']}] Expanding {len(ats_pages)} ATS listing page(s)...")
            ats_link_map = _fetch_links(tf, ats_pages[:5])
            ats_jobs = 0
            for page_links in ats_link_map.values():
                for link in page_links:
                    if is_job_url(link) and link not in seen_urls:
                        found_urls.add(link)
                        ats_jobs += 1
            logger.debug(f"  [{company['name']}] ATS expansion: {ats_jobs} additional job links")

    query = SEARCH_QUERY.format(domain=company["search_domain"])
    logger.debug(f"  [{company['name']}] Search query: {query}")
    for attempt in range(2):
        try:
            resp = tf.search.query(query, language="en")
            search_new = 0
            for r in resp.results:
                if is_job_url(r.url) and r.url not in seen_urls:
                    found_urls.add(r.url)
                    search_new += 1
            logger.debug(f"  [{company['name']}] Search: {len(resp.results)} results, {search_new} new job URLs")
            time.sleep(13)
            break
        except RateLimitError:
            logger.warning(f"  [{company['name']}] Search rate-limited — waiting 60s...")
            time.sleep(62)
        except Exception as e:
            logger.error(f"  [{company['name']}] Search error: {e}")
            time.sleep(13)
            break

    new = [
        {
            "url": u,
            "title": u.split("/")[-1].replace("-", " ").title(),
            "snippet": "",
            "company": company["name"],
            "location": company["location"],
            "region": company["region"],
        }
        for u in found_urls
    ]
    return new


def fetch_job_details(tf: TinyFish, jobs: list[dict]) -> list[dict]:
    enriched = []
    for i in range(0, len(jobs), 10):
        batch = jobs[i: i + 10]
        urls = [j["url"] for j in batch]
        logger.debug(f"  Fetching details for {len(batch)} job(s): {[j['title'][:40] for j in batch]}")
        resp = _fetch_with_ratelimit(tf, urls, format="markdown")
        if not resp:
            enriched.extend(batch)
            continue
        fetched = {r.url: r for r in resp.results}
        for job in batch:
            r = fetched.get(job["url"])
            if r and r.text:
                job["content"] = r.text[:3000]
                job["title"] = r.title or job["title"]
                logger.debug(f"    Fetched '{job['title']}' — {len(r.text)} chars")
            else:
                logger.debug(f"    No content for: {job['url']}")
            enriched.append(job)
    return enriched


def score_jobs(jobs: list[dict], resume: str, config: dict) -> list[dict]:
    if not jobs:
        return []

    jobs_text = "\n\n".join(
        f"JOB {i + 1}: {j['company']} | {j['location']}\n"
        f"Title: {j['title']}\n"
        f"{j.get('content', j.get('snippet', ''))[:800]}"
        for i, j in enumerate(jobs)
    )

    min_score = config.get("candidate", {}).get("min_score", 55)
    candidate_profile = _build_candidate_profile(config)

    prompt = SCORE_PROMPT.format(
        job_count=len(jobs),
        candidate_profile=candidate_profile,
        resume_summary=resume[:800],
        jobs_text=jobs_text,
        min_score=min_score,
    )

    messages = [
        {"role": "system", "content": _SCORE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    logger.debug(f"  Scoring {len(jobs)} job(s) via LLM (min_score={min_score})...")

    scored = None
    for attempt in range(2):
        t0 = time.time()
        try:
            raw = chat_with_llm(config, messages=messages, temperature=0.1)
            elapsed = time.time() - t0
            scored = _extract_json_array(raw)
            if scored is not None:
                logger.debug(f"  LLM OK in {elapsed:.1f}s — {len(scored)} results")
                break
            logger.warning(f"  LLM non-JSON (attempt {attempt + 1}/2, {elapsed:.1f}s): {raw[:120]!r}")
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  LLM error (attempt {attempt + 1}/2, {elapsed:.1f}s): {e}")

    # Throttle between batches to avoid OpenRouter 429s across companies
    _delay = config.get("llm_inter_request_delay", 3)
    if _delay > 0:
        time.sleep(_delay)

    if scored is None:
        logger.warning("  LLM unavailable — using rule-based fallback scoring")
        scored_objs = [_rule_based_score(j, config) for j in jobs]
        passing = [j for j in scored_objs if j.get("worth_applying")]
        logger.info(f"  Rule-based: {len(passing)}/{len(jobs)} jobs passed threshold")
        return sorted(passing, key=lambda x: x["score"], reverse=True)

    results = []
    for item in scored:
        score = item.get("score", 0)
        title = item.get("title", "?")
        reason = item.get("reason", "")
        worth = item.get("worth_applying", False)
        logger.debug(f"    [{score:3d}] {title} — {reason[:80]}")
        if not worth:
            continue
        idx = item.get("job_number", 0) - 1
        if 0 <= idx < len(jobs):
            job = jobs[idx].copy()
            job.update({
                "score": score,
                "extracted_title": title,
                "stack": item.get("stack", ""),
                "location_remote": item.get("location_remote", job["location"]),
                "reason": reason,
            })
            results.append(job)

    logger.debug(f"  {len(results)}/{len(scored)} jobs passed min_score threshold")
    return sorted(results, key=lambda x: x["score"], reverse=True)


def format_telegram_message(top_jobs: list[dict], date_str: str) -> str:
    lines = [f"<b>Job Hunt — {date_str}</b>", f"<i>{len(top_jobs)} matches found</i>\n"]
    for i, job in enumerate(top_jobs, 1):
        lines.append(
            f"<b>#{i}</b> | {job['company']} | {job.get('extracted_title', job['title'])}\n"
            f"📍 {job.get('location_remote', job['location'])}\n"
            f"🔧 {job.get('stack', 'N/A')}\n"
            f"✅ {job.get('reason', '')}\n"
            f"<a href=\"{job['url']}\">Apply</a>\n"
        )
    lines.append('Reply "apply to #N" to draft application.')
    return "\n".join(lines)


def _export_to_csv(jobs: list[dict], label: str, output_dir: Path = Path("output")) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = output_dir / f"jobs_{date_str}.csv"
    out_path.parent.mkdir(exist_ok=True)

    def _row(j: dict) -> dict:
        worth = j.get("worth_applying")
        return {
            "Company": j.get("company", ""),
            "Role": j.get("extracted_title") or j.get("title", ""),
            "Location": j.get("location_remote") or j.get("location", ""),
            "Application URL": j.get("url", ""),
            "Score (%)": j.get("score", ""),
            "Stack": j.get("stack", ""),
            "Region": j.get("region", ""),
            "Reason": j.get("reason", ""),
            "Worth Applying": "Yes" if worth else ("No" if worth is False else ""),
            "Scan Date": j.get("scan_date", ""),
        }

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for j in jobs:
            writer.writerow(_row(j))

    logger.info(f"Results exported to CSV ({label}): {out_path}")
    return out_path


def run_scan(
    config: dict,
    companies: list[dict],
    state_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    _state_dir = state_dir if state_dir is not None else Path("state")
    _output_dir = output_dir if output_dir is not None else Path("output")
    state_file = _state_dir / "seen_jobs.json"
    last_scan_file = _state_dir / "last_scan.json"
    job_history_file = _state_dir / "job_history.json"

    scan_start = time.time()
    total = len(companies)
    batch_timeout_minutes = config.get("batch_timeout_minutes", 7)
    batch_deadline = scan_start + batch_timeout_minutes * 60

    # ── Pre-create dirs and write sentinel state so artifact upload never fails ──
    _state_dir.mkdir(parents=True, exist_ok=True)
    _output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Scan started — {total} companies | state={_state_dir} output={_output_dir} | budget={batch_timeout_minutes}m ===")
    logger.info(f"Candidate: {config.get('candidate', {}).get('name', 'unknown')}")
    logger.info(f"Min score: {config.get('candidate', {}).get('min_score', 55)} | Top N: {config.get('candidate', {}).get('top_n', 5)}")
    provider = config.get("llm_provider") or "openrouter"
    model_by_provider = {
        "openrouter": config.get("openrouter_model", "default"),
        "anthropic": config.get("anthropic_model", "default"),
        "claude_cli": config.get("claude_cli_model") or "claude default",
    }
    logger.info(f"LLM provider: {provider} | Model: {model_by_provider.get(provider, 'default')}")

    try:
        tf = TinyFish(api_key=config["tinyfish_api_key"])
        logger.debug("TinyFish client initialised")
    except Exception as e:
        logger.error(f"TinyFish init error: {e}")
        return

    resume_path = Path(config.get("candidate", {}).get("resume_path", "resume/YOUR_RESUME.md"))
    resume = resume_path.read_text()
    logger.debug(f"Resume loaded: {resume_path} ({len(resume)} chars)")

    min_score = config.get("candidate", {}).get("min_score", 55)
    top_n = config.get("candidate", {}).get("top_n", 5)

    state = load_state(state_file)
    seen_urls: set = set(state.get("seen_urls", []))
    logger.info(f"State loaded — {len(seen_urls)} previously seen URLs")

    # Write a minimal sentinel immediately so the file exists even if the batch times out
    if not state_file.exists():
        save_state({"seen_urls": list(seen_urls), "last_scan": None}, state_file)

    all_scored_jobs: list[dict] = []
    errors: list[str] = []
    companies_scanned = 0
    companies_with_jobs = 0
    companies_skipped_budget = 0

    for idx, company in enumerate(companies, 1):
        # ── Time budget: stop before GitHub cancels the runner ──────────────
        elapsed_m = (time.time() - scan_start) / 60
        remaining_m = (batch_deadline - time.time()) / 60
        if time.time() > batch_deadline:
            companies_skipped_budget = total - idx + 1
            logger.warning(
                f"Time budget reached at {elapsed_m:.1f}m — "
                f"stopping with {companies_skipped_budget} company/companies remaining"
            )
            break

        logger.info(f"[{idx}/{total}] {company['name']} (budget: {remaining_m:.1f}m left)...")
        try:
            new_jobs = discover_job_urls(tf, company, seen_urls)
            if not new_jobs:
                logger.info("  No new job URLs found")
                companies_scanned += 1
                # Incremental state save so seen_urls persists even if we time out later
                state["seen_urls"] = list(seen_urls)
                save_state(state, state_file)
                continue

            # Pre-filter on URL-slug titles before spending TinyFish fetch quota
            new_jobs, pre_discarded = _prefilter_by_title(new_jobs)
            if pre_discarded:
                logger.info(f"  Pre-filter: discarded {pre_discarded} (seniority/function)")

            # Hard cap — sort is not yet possible (no dates pre-fetch), take first N
            if len(new_jobs) > _MAX_JOBS_PER_COMPANY:
                logger.info(f"  Capping {len(new_jobs)} → {_MAX_JOBS_PER_COMPANY} jobs")
                new_jobs = new_jobs[:_MAX_JOBS_PER_COMPANY]

            if not new_jobs:
                logger.info("  No jobs remaining after pre-filter")
                companies_scanned += 1
                state["seen_urls"] = list(seen_urls)
                save_state(state, state_file)
                continue

            logger.info(f"  {len(new_jobs)} job(s) to fetch...")
            new_jobs = fetch_job_details(tf, new_jobs)
            seen_urls.update(j["url"] for j in new_jobs)

            # Post-filter on real title + fetched content
            new_jobs, post_discarded = _prefilter_by_content(new_jobs)
            if post_discarded:
                logger.info(f"  Post-filter: discarded {post_discarded} more after content fetch")

            if not new_jobs:
                logger.info("  No jobs remaining after content filter")
                companies_scanned += 1
                state["seen_urls"] = list(seen_urls)
                save_state(state, state_file)
                continue

            logger.info(f"  Scoring {len(new_jobs)} job(s) in batches of 10...")
            scored: list[dict] = []
            for i in range(0, len(new_jobs), 10):
                batch = new_jobs[i: i + 10]
                logger.debug(f"  Scoring batch {i // 10 + 1} ({len(batch)} jobs)...")
                batch_scored = score_jobs(batch, resume, config)
                scored.extend(batch_scored)

            if scored:
                all_scored_jobs.extend(scored)
                companies_with_jobs += 1
                titles = [j.get("extracted_title") or j.get("title", "?") for j in scored[:3]]
                logger.info(f"  {len(scored)} job(s) scored: {', '.join(titles)}{' ...' if len(scored) > 3 else ''}")

            companies_scanned += 1

            # Incremental state save after every company so a timeout doesn't lose progress
            state["seen_urls"] = list(seen_urls)
            save_state(state, state_file)

        except Exception as company_err:
            msg = f"❌ {company['name']}: {company_err}"
            errors.append(msg)
            logger.error(f"  Company scan failed: {company_err}")
            continue

    state["seen_urls"] = list(seen_urls)
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state, state_file)

    # ── Batch summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - scan_start
    logger.info(
        f"=== Batch summary: {companies_scanned}/{total} companies scanned | "
        f"{companies_with_jobs} with hits | {len(all_scored_jobs)} jobs scored | "
        f"{elapsed_total / 60:.1f}m elapsed"
        + (f" | {companies_skipped_budget} skipped (budget)" if companies_skipped_budget else "")
        + " ==="
    )
    if errors:
        logger.warning(f"  Errors ({len(errors)}): " + " | ".join(errors))

    top_jobs = sorted(
        [j for j in all_scored_jobs if j.get("score", 0) >= min_score],
        key=lambda x: x.get("score", 0), reverse=True
    )[:top_n]

    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for job in all_scored_jobs:
        job["scan_date"] = scan_date

    last_scan_file.parent.mkdir(exist_ok=True)
    last_scan_file.write_text(json.dumps(all_scored_jobs, indent=2))
    logger.debug(f"Last scan saved: {len(all_scored_jobs)} total jobs → {last_scan_file}")

    history: list[dict] = []
    if job_history_file.exists():
        try:
            history = json.loads(job_history_file.read_text())
        except Exception:
            history = []
    existing_urls = {j["url"] for j in history}
    new_entries = [j for j in all_scored_jobs if j["url"] not in existing_urls]
    history.extend(new_entries)
    job_history_file.write_text(json.dumps(history, indent=2))
    logger.debug(f"Job history updated: +{len(new_entries)} new entries ({len(history)} total)")

    elapsed = time.time() - scan_start
    logger.info(
        f"=== Scan complete — {companies_scanned}/{total} companies, "
        f"{len(all_scored_jobs)} jobs found, {len(top_jobs)} top matches "
        f"({elapsed / 60:.1f} min) ==="
    )

    if top_jobs:
        logger.info("Top matches:")
        for j in top_jobs:
            logger.info(f"  [{j.get('score', '?'):3}] {j.get('extracted_title') or j.get('title')} @ {j['company']} — {j.get('reason', '')[:80]}")

    date_str = datetime.now().strftime("%d %b %Y")
    tg = config.get("telegram", {})
    telegram_configured = bool(tg.get("token") and tg.get("chat_id"))

    # Always persist results to CSV when there are scored jobs — this is the
    # durable record regardless of whether Telegram is configured.
    csv_path = _export_to_csv(all_scored_jobs, "scan results", _output_dir) if all_scored_jobs else None

    if errors and telegram_configured:
        error_msg = f"<b>Job Hunt Errors — {date_str}</b>\n" + "\n".join(errors)
        send_telegram(tg["token"], tg["chat_id"], error_msg)

    if not top_jobs:
        logger.info("No matching jobs found today.")
        if telegram_configured:
            msg = f"<b>Job Hunt — {date_str}</b>\nNo new matches today."
            send_telegram(tg["token"], tg["chat_id"], msg)
        return

    msg = format_telegram_message(top_jobs, date_str)
    logger.info("\n" + msg)

    # Telegram is an optional notification on top of the CSV. When it's not
    # configured we simply skip it — no error, the CSV already holds the results.
    if telegram_configured:
        sent = send_telegram(tg["token"], tg["chat_id"], msg)
        if sent:
            logger.info(f"Telegram notification sent. Results also saved to CSV: {csv_path}")
        else:
            logger.warning(f"Telegram send failed — results saved to CSV: {csv_path}")
    else:
        logger.info(f"Telegram not configured — results saved to CSV: {csv_path}")
        logger.info("Add telegram.token and telegram.chat_id to config.json to enable notifications.")
