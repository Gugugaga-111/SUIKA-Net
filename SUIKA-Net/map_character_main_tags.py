#!/usr/bin/env python3
"""
Map Touhou character list to Pixiv main tags.

Heuristic used:
1. Fetch autocomplete candidates from Pixiv app-api v2/search/autocomplete.
2. Validate top candidates by sampling exact-match search results.
3. Pick the best candidate prioritizing:
   - tags that strongly match the character name
   - Touhou-related sample hits
   - autocomplete rank (proxy of popularity)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixivpy3 import AppPixivAPI


AUTOCOMPLETE_URL = "https://app-api.pixiv.net/v2/search/autocomplete"
TOUHOU_TAG_PREFIX = "東方"


@dataclass
class Candidate:
    tag: str
    query: str
    auto_rank: int
    sample_size: int = 0
    touhou_hits: int = 0
    char_hits: int = 0
    score: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve Pixiv main tags for top character list.")
    parser.add_argument(
        "--refresh-token",
        default=None,
        help="Pixiv refresh token, or use PIXIV_REFRESH_TOKEN env var.",
    )
    parser.add_argument(
        "--input-csv",
        default="/path/to/bs1/character_top100.csv",
        help="Input CSV with columns index,rank,name.",
    )
    parser.add_argument(
        "--output-csv",
        default="/path/to/bs1/character_top100_main_tags.csv",
        help="Output mapping CSV path.",
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/bs1/character_top100_main_tags_debug.json",
        help="Output debug JSON path.",
    )
    parser.add_argument(
        "--eval-candidates",
        type=int,
        default=6,
        help="How many autocomplete candidates to sample per character.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep seconds between API requests to reduce rate-limit risk.",
    )
    return parser.parse_args()


def require_refresh_token(cli_value: str | None) -> str:
    token = cli_value or os.getenv("PIXIV_REFRESH_TOKEN")
    if not token:
        raise ValueError("Refresh token is required via --refresh-token or PIXIV_REFRESH_TOKEN.")
    return token


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    # Remove common separators/whitespaces; keep Japanese long vowel mark "ー".
    t = re.sub(r"[\s·・･\-‐‑‒–—―~〜_/]+", "", t)
    return t.lower()


def cleanup_query(text: str) -> str:
    # For autocomplete query, remove spaces and punctuation that often split character full names.
    t = unicodedata.normalize("NFKC", text).strip()
    t = re.sub(r"[\s·・･\-‐‑‒–—―~〜_/]+", "", t)
    return t


def is_touhou_tag(tag_name: str) -> bool:
    n = unicodedata.normalize("NFKC", tag_name)
    return n.startswith(TOUHOU_TAG_PREFIX)


def safe_parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def build_query_variants(character_name: str) -> list[str]:
    q1 = character_name.strip()
    q2 = re.sub(r"\s+", "", q1)
    q3 = cleanup_query(q1)
    parts = [p for p in re.split(r"[\s·・･\-‐‑‒–—―~〜_/]+", q1) if p]
    queries: list[str] = []
    for q in (q1, q2, q3, *parts):
        if q and q not in queries:
            queries.append(q)
    return queries


def call_autocomplete(api: AppPixivAPI, query: str, sleep_sec: float) -> tuple[list[str], str]:
    if not query:
        return [], "empty_query"
    params = {"word": query}
    try:
        resp = api.no_auth_requests_call("GET", AUTOCOMPLETE_URL, params=params, req_auth=True)
    except Exception as err:
        return [], f"request_error: {err}"

    payload = safe_parse_json(resp.text)
    tags = payload.get("tags", []) or []
    out: list[str] = []
    for item in tags:
        name = str(item.get("name", "")).strip()
        if name:
            out.append(name)
    error_obj = payload.get("error", {})
    err_msg = ""
    if isinstance(error_obj, dict):
        err_msg = str(error_obj.get("message", "")).strip()
    if not err_msg and not out:
        err_msg = "no_tags"
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return out, err_msg


def collect_candidates(
    api: AppPixivAPI, character_name: str, sleep_sec: float
) -> tuple[list[Candidate], list[str], dict[str, Any]]:
    queries = build_query_variants(character_name)
    seen: set[str] = set()
    candidates: list[Candidate] = []
    autocomplete_debug: dict[str, Any] = {}
    for query in queries:
        tags, error = call_autocomplete(api, query, sleep_sec=sleep_sec)
        autocomplete_debug[query] = {"candidates": tags[:10], "error": error}
        for idx, tag in enumerate(tags, start=1):
            key = normalize(tag)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(Candidate(tag=tag, query=query, auto_rank=idx))
    return candidates, queries, autocomplete_debug


def eval_candidate(
    api: AppPixivAPI,
    character_name: str,
    cand: Candidate,
    sleep_sec: float,
) -> Candidate:
    char_norm = normalize(character_name)
    try:
        result = api.search_illust(
            word=cand.tag,
            search_target="exact_match_for_tags",
            sort="date_desc",
            search_ai_type=0,
        )
    except Exception:
        return cand

    illusts = getattr(result, "illusts", []) or []
    cand.sample_size = len(illusts)
    touhou_hits = 0
    char_hits = 0
    for illust in illusts:
        tags = getattr(illust, "tags", []) or []
        names = [str(getattr(t, "name", "")).strip() for t in tags]
        if any(is_touhou_tag(x) for x in names):
            touhou_hits += 1
        normalized_names = [normalize(x) for x in names if x]
        if any((char_norm == n) or (char_norm in n) or (n in char_norm) for n in normalized_names if n):
            char_hits += 1

    cand.touhou_hits = touhou_hits
    cand.char_hits = char_hits
    tag_norm = normalize(cand.tag)
    match_level = 0
    if tag_norm == char_norm:
        match_level = 3
    elif char_norm and char_norm in tag_norm:
        match_level = 2
    elif tag_norm and tag_norm in char_norm:
        match_level = 1
    # Higher is better.
    # Priority: tag-name similarity > character sample match > touhou relevance > sample stats > autocomplete rank.
    cand.score = (
        match_level,
        1 if char_hits > 0 else 0,
        1 if touhou_hits > 0 else 0,
        touhou_hits,
        char_hits * 10 - cand.auto_rank,
    )
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return cand


def choose_main_tag(character_name: str, candidates: list[Candidate]) -> tuple[str, list[Candidate], str]:
    if not candidates:
        fallback = re.sub(r"\s+", "", character_name.strip())
        return fallback, [], "fallback_no_candidates"

    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    best = sorted_cands[0]
    if best.score == (0, 0, 0, 0, 0):
        # No eval signal, fallback to top autocomplete.
        return candidates[0].tag, sorted_cands, "autocomplete_rank_only"
    return best.tag, sorted_cands, "evaluated"


def main() -> int:
    args = parse_args()
    try:
        refresh_token = require_refresh_token(args.refresh_token)
    except ValueError as err:
        print(f"[ERROR] {err}")
        return 1

    input_path = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("name"):
                rows.append(row)

    api = AppPixivAPI()
    api.set_accept_language("zh-cn")
    api.auth(refresh_token=refresh_token)

    result_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        idx = row.get("index", str(i))
        rank = row.get("rank", str(i))
        name = row["name"].strip()

        cands, query_variants, autocomplete_debug = collect_candidates(api, name, sleep_sec=args.sleep)
        source = "autocomplete"
        if not cands:
            # If autocomplete is temporarily unavailable/rate-limited, synthesize
            # conservative candidates from full-name variants to avoid blind fallback.
            synth_seen: set[str] = set()
            synth: list[Candidate] = []
            for q in query_variants[:3]:
                nq = normalize(q)
                if not nq or len(nq) < 2 or nq in synth_seen:
                    continue
                synth_seen.add(nq)
                synth.append(Candidate(tag=q, query="synthetic", auto_rank=999 + len(synth)))
            cands = synth
            if cands:
                source = "synthetic"
        eval_num = max(1, min(args.eval_candidates, len(cands)))
        evaled: list[Candidate] = []
        for cand in cands[:eval_num]:
            evaled.append(eval_candidate(api, name, cand, sleep_sec=args.sleep))
        # Keep unevaluated tails with default score for debug transparency.
        evaled.extend(cands[eval_num:])

        main_tag, sorted_cands, method = choose_main_tag(name, evaled)
        if method == "evaluated" and source == "synthetic":
            method = "evaluated_synthetic"
        elif method == "fallback_no_candidates" and source == "synthetic":
            method = "fallback_synthetic"
        top = sorted_cands[0] if sorted_cands else None

        result_rows.append(
            {
                "index": idx,
                "rank": rank,
                "name": name,
                "main_tag": main_tag,
                "method": method,
                "auto_query": top.query if top else "",
                "auto_rank": top.auto_rank if top else "",
                "sample_size": top.sample_size if top else "",
                "touhou_hits_sample": top.touhou_hits if top else "",
                "char_hits_sample": top.char_hits if top else "",
            }
        )
        debug_rows.append(
            {
                "index": idx,
                "rank": rank,
                "name": name,
                "main_tag": main_tag,
                "method": method,
                "candidates_sorted": [
                    {
                        "tag": c.tag,
                        "query": c.query,
                        "auto_rank": c.auto_rank,
                        "sample_size": c.sample_size,
                        "touhou_hits": c.touhou_hits,
                        "char_hits": c.char_hits,
                        "score": c.score,
                    }
                    for c in sorted_cands[:20]
                ],
                "query_variants": query_variants,
                "autocomplete_debug": autocomplete_debug,
            }
        )
        print(f"[{i:03d}/{total:03d}] {name} -> {main_tag} ({method})")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "rank",
                "name",
                "main_tag",
                "method",
                "auto_query",
                "auto_rank",
                "sample_size",
                "touhou_hits_sample",
                "char_hits_sample",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    output_json.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote: {output_csv}")
    print(f"[DONE] Wrote: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
