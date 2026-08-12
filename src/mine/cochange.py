"""Build a co-change (evolutionary coupling) graph and cluster it into feature groups.

Files that repeatedly change in the same commit are coupled in practice regardless of
what the directory tree says. Clustering that coupling gives de-facto feature/epic
groups, derived mechanically -- no model reads any code.

Those clusters do real work downstream: arXiv 2508.10074 needed GPT-4o-mini to discard
72.8% of semantically incoherent edit sequences before training a next-edit model. Here,
requiring that a commit's hunks fall inside one cluster performs the same filtering by
construction, which is what keeps this pipeline LLM-free.

Weighting notes:
  - each commit contributes exp(-age_days/HALFLIFE) rather than 1, so a module's coupling
    reflects how it is edited now, not how it was edited two years ago
  - Jaccard, not raw counts, so a file touched in every commit does not couple to everything
  - clustering is per-repo; cross-repo coupling is not meaningful

Usage:
    python src/mine/cochange.py [--min-pair 3] [--min-jaccard 0.15]
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities

DATA = Path(__file__).parent.parent.parent / "data" / "mined"
COMMITS = DATA / "commits.jsonl"
OUTPUT = DATA / "clusters.json"

HALFLIFE_DAYS = 180.0
DAY = 86400.0


def load_commits(path):
    by_repo = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            by_repo[rec["repo"]].append(rec)
    return by_repo


def common_prefix(paths):
    if not paths:
        return ""
    parts = [p.split("/")[:-1] for p in paths]
    if not parts or not parts[0]:
        return ""
    pref = parts[0]
    for other in parts[1:]:
        i = 0
        while i < len(pref) and i < len(other) and pref[i] == other[i]:
            i += 1
        pref = pref[:i]
        if not pref:
            break
    return "/".join(pref)


def label_cluster(files, commits_for_file):
    """Name a cluster from the modal conventional-commit scope, else its path prefix."""
    scopes = Counter()
    for f in files:
        for c in commits_for_file.get(f, ()):
            if c.get("scope"):
                scopes[c["scope"]] += 1
    if scopes:
        top, n = scopes.most_common(1)[0]
        if n >= 2:
            return top
    prefix = common_prefix(files)
    if prefix:
        return prefix.split("/")[-1] or prefix
    return re.sub(r"\.[^.]+$", "", Path(sorted(files)[0]).name)


def build_repo_clusters(commits, min_pair, min_jaccard, now):
    file_w = defaultdict(float)
    pair_w = defaultdict(float)
    pair_n = Counter()
    commits_for_file = defaultdict(list)

    for c in commits:
        decay = math.exp(-max(0.0, (now - c["ts"]) / DAY) / HALFLIFE_DAYS)
        files = c["files"]
        for f in files:
            file_w[f] += decay
            commits_for_file[f].append(c)
        if len(files) < 2:
            continue
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                key = (files[i], files[j])
                pair_w[key] += decay
                pair_n[key] += 1

    g = nx.Graph()
    for (a, b), w in pair_w.items():
        if pair_n[(a, b)] < min_pair:
            continue
        denom = file_w[a] + file_w[b] - w
        if denom <= 0:
            continue
        jac = w / denom
        if jac < min_jaccard:
            continue
        g.add_edge(a, b, weight=jac)

    if g.number_of_nodes() == 0:
        return [], g, commits_for_file

    communities = louvain_communities(g, weight="weight", seed=42)
    return communities, g, commits_for_file


def main():
    ap = argparse.ArgumentParser(description="Cluster files by co-change coupling")
    ap.add_argument("--commits", default=str(COMMITS))
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--min-pair", type=int, default=3,
                    help="minimum raw co-occurrence count for an edge")
    ap.add_argument("--min-jaccard", type=float, default=0.15)
    ap.add_argument("--min-cluster", type=int, default=2)
    args = ap.parse_args()

    by_repo = load_commits(args.commits)
    now = max(c["ts"] for cs in by_repo.values() for c in cs)
    print(f"{sum(len(v) for v in by_repo.values()):,} commits across {len(by_repo)} repos")

    out = {"params": {"min_pair": args.min_pair, "min_jaccard": args.min_jaccard,
                      "halflife_days": HALFLIFE_DAYS}, "repos": {}}
    tot_clusters = tot_files = 0
    sizes = []

    for repo, commits in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        communities, g, commits_for_file = build_repo_clusters(
            commits, args.min_pair, args.min_jaccard, now)
        clusters = []
        for idx, comm in enumerate(communities):
            files = sorted(comm)
            if len(files) < args.min_cluster:
                continue
            sub = g.subgraph(files)
            cohesion = (sum(d["weight"] for *_, d in sub.edges(data=True)) /
                        max(1, sub.number_of_edges()))
            clusters.append({
                "cluster_id": f"{repo}:{idx}",
                "label": label_cluster(files, commits_for_file),
                "files": files,
                "n_files": len(files),
                "cohesion": round(cohesion, 4),
            })
        if not clusters:
            continue
        clusters.sort(key=lambda c: -c["n_files"])
        out["repos"][repo] = {"n_commits": len(commits), "clusters": clusters}
        tot_clusters += len(clusters)
        tot_files += sum(c["n_files"] for c in clusters)
        sizes += [c["n_files"] for c in clusters]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    sizes.sort()
    print(f"\n{'='*60}\nCO-CHANGE CLUSTERING COMPLETE\n{'='*60}")
    print(f"Repos with clusters: {len(out['repos'])}")
    print(f"Clusters: {tot_clusters:,}   files covered: {tot_files:,}")
    if sizes:
        print(f"Cluster size: median={sizes[len(sizes)//2]}  p90={sizes[int(len(sizes)*.9)]}  max={sizes[-1]}")
    print(f"Output: {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")

    for repo in list(out["repos"])[:3]:
        print(f"\n  {repo}:")
        for c in out["repos"][repo]["clusters"][:3]:
            print(f"    [{c['label']}] {c['n_files']} files, cohesion {c['cohesion']}")
            for f in c["files"][:3]:
                print(f"        {f}")


if __name__ == "__main__":
    main()
