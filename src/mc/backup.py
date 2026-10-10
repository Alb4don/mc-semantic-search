"""Story 5.3 — collection snapshot backup & restore.

Nightly Mode A snapshots (Story 5.1 cron) are taken through the Qdrant
snapshots HTTP API — never by copying the live storage dir. Snapshot files
land in the qdrant snapshots volume: ``./snapshots`` on the host, mounted
at ``/qdrant/snapshots`` inside the qdrant container (docker-compose.yml).

Restore caveat (verified against qdrant v1.12.4): Qdrant reads the
snapshot file from ITS OWN filesystem, so the file must sit inside the
qdrant container's snapshots dir and the recover call must pass a
``file://`` URL pointing there. A plain server path is rejected with
``relative URL without a base``.

CLI (mirrors the cron entry in crontab.example):
    python -m mc.backup create [--keep 7] [--collection NAME]
    python -m mc.backup restore SNAPSHOT_FILE --to TARGET_COLLECTION
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from mc.config import COLLECTION_NAME, QDRANT_URL, SNAPSHOTS_DIR
from mc.qdrant import client_kwargs

logger = logging.getLogger(__name__)

#: Path of the snapshots volume inside the qdrant container
#: (docker-compose.yml maps ./snapshots -> /qdrant/snapshots).
CONTAINER_SNAPSHOTS_DIR = "/qdrant/snapshots"

#: Rotation policy: keep the last N snapshots per collection.
DEFAULT_KEEP = 7


def default_client(url: str | None = None) -> QdrantClient:
    """Client for the configured qdrant (``config.QDRANT_URL``)."""
    return QdrantClient(url=url or QDRANT_URL, **client_kwargs(timeout=120))


def snapshot_dir(collection: str, snapshots_path: Path | None = None) -> Path:
    """Per-collection snapshot dir (qdrant nests ``<snapshots>/<collection>/``)."""
    return Path(snapshots_path or SNAPSHOTS_DIR) / collection


def create_snapshot(
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    snapshots_path: Path | None = None,
    keep: int | None = DEFAULT_KEEP,
) -> Path:
    """Snapshot `collection` via the API and return the snapshot file path.

    The file physically lands in the qdrant container's snapshots volume
    (``./snapshots/<collection>/<name>.snapshot`` on the host). With
    ``keep=N`` (default 7) older snapshots of the same collection are
    deleted through the API — the files are root-owned inside the volume,
    so API deletion works regardless of the caller's uid.
    """
    client = client or default_client()
    desc = client.create_snapshot(collection_name=collection, wait=True)
    path = snapshot_dir(collection, snapshots_path) / desc.name
    logger.info("snapshot created: %s (%s bytes)", path, desc.size)
    if keep is not None:
        _rotate_via_api(client, collection, keep)
    return path


def _rotate_via_api(client: QdrantClient, collection: str, keep: int) -> None:
    """Keep the newest `keep` snapshots of `collection`, API-delete the rest."""
    snaps = client.list_snapshots(collection_name=collection)
    ordered = sorted(snaps, key=lambda s: str(s.creation_time or ""), reverse=True)
    for desc in ordered[keep:]:
        logger.info("rotation: deleting old snapshot %s", desc.name)
        client.delete_snapshot(
            collection_name=collection, snapshot_name=desc.name, wait=True
        )


def rotate(snapshots_dir: Path, keep: int = DEFAULT_KEEP) -> list[Path]:
    """Delete the oldest ``*.snapshot`` files beyond `keep` (newest survive).

    Filesystem-level rotation for manual/drill use; :func:`create_snapshot`
    rotates through the API instead (see the root-owned-volume note there).
    Companion ``*.snapshot.checksum`` files are removed with their snapshot.
    """
    files = sorted(
        Path(snapshots_dir).glob("*.snapshot"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed: list[Path] = []
    for old in files[keep:]:
        try:
            old.unlink()
            Path(f"{old}.checksum").unlink(missing_ok=True)
            removed.append(old)
        except OSError as exc:
            logger.warning("rotation: could not delete %s: %s", old, exc)
    return removed


def _container_location(
    snapshot_path: Path,
    snapshots_path: Path | None = None,
    container_snapshots_dir: str = CONTAINER_SNAPSHOTS_DIR,
) -> str:
    """Map a host snapshot path to a ``file://`` URL the qdrant server accepts."""
    path = Path(snapshot_path).resolve()
    base = Path(snapshots_path or SNAPSHOTS_DIR).resolve()
    try:
        rel = path.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"{path} is not under the snapshots dir {base} — the qdrant "
            "container can only read snapshots inside its snapshots volume "
            f"({container_snapshots_dir})"
        ) from exc
    return f"file://{container_snapshots_dir}/{rel}"


def restore(
    snapshot_path: Path,
    target_collection: str,
    client: QdrantClient | None = None,
    *,
    snapshots_path: Path | None = None,
    container_snapshots_dir: str = CONTAINER_SNAPSHOTS_DIR,
) -> None:
    """Recover `target_collection` from a snapshot file (Story 5.3 drill).

    The target must NOT already exist (enforced below — a silent overwrite
    of the live collection is exactly what this must prevent). The snapshot
    file is read by the qdrant server itself, so it must be inside the
    container's snapshots volume; the recover call uses a ``file://`` URL there.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"snapshot file not found: {path}")
    location = _container_location(path, snapshots_path, container_snapshots_dir)
    client = client or default_client()
    if client.collection_exists(target_collection):
        raise ValueError(
            f"target collection {target_collection!r} already exists — restore "
            "never overwrites an existing collection (drop it or pick a new --to)"
        )
    logger.info("recovering %s from %s", target_collection, location)
    ok = client.recover_snapshot(
        collection_name=target_collection, location=location, wait=True
    )
    if not ok:
        raise RuntimeError(f"qdrant refused to recover {target_collection}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mc.backup", description="Qdrant snapshot backup/restore (Story 5.3)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser(
        "create", help="snapshot a collection and rotate old snapshots"
    )
    p_create.add_argument("--collection", default=COLLECTION_NAME)
    p_create.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    p_create.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="host snapshots base dir (default config.SNAPSHOTS_DIR)",
    )
    p_create.add_argument("--url", default=None, help="qdrant url override")

    p_restore = sub.add_parser("restore", help="recover a collection from a snapshot")
    p_restore.add_argument("snapshot", type=Path)
    p_restore.add_argument(
        "--to",
        dest="target",
        required=True,
        help="target collection name (must not exist)",
    )
    p_restore.add_argument("--snapshots-dir", type=Path, default=None)
    p_restore.add_argument("--url", default=None)

    args = ap.parse_args(argv)
    client = default_client(args.url)
    if args.cmd == "create":
        path = create_snapshot(
            client=client,
            collection=args.collection,
            snapshots_path=args.snapshots_dir,
            keep=args.keep,
        )
        print(path)
    else:
        restore(
            args.snapshot,
            args.target,
            client=client,
            snapshots_path=args.snapshots_dir,
        )
        print(f"recovered {args.target} from {args.snapshot}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
