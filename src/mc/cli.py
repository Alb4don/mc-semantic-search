"""`mc` command line: every operator task scriptable by cron (Story 0.3).

Subcommands are wired to their story implementations as the epic proceeds;
the skeleton exits non-zero for unknown commands and prints usage.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_backfill(args: argparse.Namespace) -> int:
    from mc.ingest.backfill import run_backfill

    return run_backfill(
        from_db=args.from_db,
        batch_size=args.batch_size,
        limit=args.limit,
    )


def _cmd_update(args: argparse.Namespace) -> int:
    from mc.ingest.update import run_update

    return run_update(dry_run=args.dry_run)


def _cmd_push(args: argparse.Namespace) -> int:
    from mc.ingest.push import run_push

    return run_push()


def _cmd_apply(args: argparse.Namespace) -> int:
    from mc.ingest.apply import run_apply

    return run_apply()


def _cmd_stats_backfill(args: argparse.Namespace) -> int:
    from mc.ingest.stats_backfill import run_stats_backfill

    report = run_stats_backfill(
        fetch_missing=args.fetch_missing,
        limit=args.limit,
    )
    print(report["summary"], end="")
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    from mc.ingest.reindex import run_reindex

    return run_reindex(model=args.model)


def _cmd_eval(args: argparse.Namespace) -> int:
    from mc.eval import format_report, load_suite, run_eval

    report = run_eval(load_suite(args.suite), limit=args.limit)
    text = format_report(report)
    print(text, end="")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    # Story 5.4 DoD: hybrid must not regress verbatim queries vs sparse-only
    return 0 if report.mrr_hybrid >= report.mrr_sparse else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from mc.config import API_PORT

    uvicorn.run("mc.api.app:app", host="0.0.0.0", port=API_PORT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mc",
        description="Morning Crypto semantic search operations CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backfill", help="embed + index all chunks from the legacy DB")
    p.add_argument("--from-db", required=True, help="path to mc_transcripts.db")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_backfill)

    p = sub.add_parser("update", help="daily ingestion: discover→download→chunk→embed→upsert")
    p.add_argument("--dry-run", action="store_true", help="verify cookies, don't ingest")
    p.set_defaults(func=_cmd_update)

    p = sub.add_parser("push", help="ship the daily increment (Mode B)")
    p.set_defaults(func=_cmd_push)

    p = sub.add_parser("apply", help="embed + upsert queued increments (Mode B server)")
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser(
        "stats-backfill",
        help="backfill archive stats for already-indexed videos (no fetch by default)",
    )
    p.add_argument(
        "--fetch-missing",
        action="store_true",
        help="yt-dlp --dump-single-json for videos still lacking a duration "
        "(upgrades the cache once; rate-limited, resumable)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap videos considered this run")
    p.set_defaults(func=_cmd_stats_backfill)

    p = sub.add_parser("reindex", help="rebuild into a new collection and swap alias")
    p.add_argument("--model", required=True, help="dense model id to reindex with")
    p.set_defaults(func=_cmd_reindex)

    p = sub.add_parser("eval", help="sparse-only vs hybrid MRR on known quotes")
    p.add_argument("--suite", required=True, help="path to quotes.yaml")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default=None, help="write the markdown report here")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="run the search API")
    p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
