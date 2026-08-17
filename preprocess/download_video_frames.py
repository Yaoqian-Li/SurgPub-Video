#!/usr/bin/env python3
import argparse
import csv
import html
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
 

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".m4v",
    ".avi",
    ".flv",
    ".ts",
    ".m2ts",
}


def _require_cmd(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise SystemExit(f"Missing required command: {cmd!r}. Install it and try again.")
    return path


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _pick_latest_video(directory: Path, *, since_epoch: float) -> Optional[Path]:
    candidates: List[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime >= since_epoch:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def download_with_ytdlp(url: str, output_dir: Path, ytdlp_cmd: str, *, outtmpl: Optional[str] = None) -> Path:
    # Allow passing a local file path instead of a URL.
    as_path = Path(url).expanduser()
    if as_path.exists() and as_path.is_file():
        return as_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ask yt-dlp to print the final filepath after move; keep output minimal for parsing.
    # outtmpl is limited to avoid weird path issues; adjust if you want.
    outtmpl = outtmpl or "%(title).200B_%(id)s.%(ext)s"
    t0 = time.time()
    cp = _run(
        [
            ytdlp_cmd,
            "--no-playlist",
            "--no-progress",
            "--quiet",
            "--print",
            "after_move:filepath",
            "-P",
            str(output_dir),
            "-o",
            outtmpl,
            url,
        ]
    )

    stdout = (cp.stdout or "").strip()
    if cp.returncode == 0 and stdout:
        # yt-dlp may print multiple lines; filepath should be last line in our usage.
        candidate = Path(stdout.splitlines()[-1]).expanduser()
        if candidate.exists():
            return candidate
        # If yt-dlp printed a relative path, resolve relative to output_dir.
        candidate2 = (output_dir / candidate).resolve()
        if candidate2.exists():
            return candidate2

    # Fallback: pick the newest video file created/modified since we started.
    latest = _pick_latest_video(output_dir, since_epoch=t0 - 2)
    if latest and latest.exists():
        return latest

    msg = "yt-dlp download failed."
    if cp.stderr.strip():
        msg += f"\n{cp.stderr.strip()}"
    raise SystemExit(msg)


def clean_existing_frames(frames_dir: Path) -> int:
    removed = 0
    for p in frames_dir.glob("frame_*.png"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def extract_frames_1fps_ffmpeg(video_path: Path, frames_dir: Path, ffmpeg_cmd: str) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / "frame_%05d.png")

    cp = _run(
        [
            ffmpeg_cmd,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1",
            "-start_number",
            "0",
            out_pattern,
        ],
        cwd=video_path.parent,
    )
    if cp.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{cp.stderr.strip() or cp.stdout.strip()}")


def extract_frames_1fps_gst(video_path: Path, frames_dir: Path, gst_launch_cmd: str) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / "frame_%05d.png")

    cp = _run(
        [
            gst_launch_cmd,
            "-e",
            "-q",
            "filesrc",
            f"location={str(video_path)}",
            "!",
            "decodebin",
            "!",
            "videoconvert",
            "!",
            "videorate",
            "!",
            "video/x-raw,framerate=1/1",
            "!",
            "pngenc",
            "!",
            "multifilesink",
            f"location={out_pattern}",
            "start-index=0",
        ],
        cwd=video_path.parent,
    )
    if cp.returncode != 0:
        raise SystemExit(f"gstreamer (gst-launch-1.0) failed:\n{cp.stderr.strip() or cp.stdout.strip()}")


def _sanitize_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty id")
    # Keep directory name safe and predictable.
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))


def _normalize_url(url: str) -> str:
    url = html.unescape((url or "").strip())
    if url.startswith("//"):
        return "https:" + url
    return url


def _iter_files_check_rows(
    csv_path: Path, *, id_col: str, url_col: str, state_col: Optional[str], require_success: bool
) -> Iterable[Tuple[str, str, Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{csv_path} missing header row.")
        for col in (id_col, url_col):
            if col not in reader.fieldnames:
                raise SystemExit(f"{csv_path} missing required column: {col!r}")
        if require_success and state_col and state_col not in reader.fieldnames:
            raise SystemExit(f"{csv_path} missing state column: {state_col!r}")

        for row in reader:
            raw_id = row.get(id_col, "")
            raw_url = row.get(url_col, "")
            if not raw_id or not raw_url:
                continue
            if require_success and state_col:
                if (row.get(state_col, "") or "").strip().lower() != "success":
                    continue
            yield _sanitize_id(raw_id), _normalize_url(raw_url), row


def _extract_frames(video_path: Path, frames_dir: Path, *, backend: str, ffmpeg_cmd: str, gst_launch_cmd: str) -> None:
    if backend == "ffmpeg":
        extract_frames_1fps_ffmpeg(video_path, frames_dir, ffmpeg_cmd)
        return
    if backend == "gst":
        extract_frames_1fps_gst(video_path, frames_dir, gst_launch_cmd)
        return
    raise ValueError(f"Unknown backend: {backend!r}")


def batch_from_files_check(args) -> None:
    csv_path = Path(args.csv).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    ytdlp_cmd = None
    ffmpeg_cmd = None
    gst_cmd = None
    backend = args.backend
    if not args.dry_run:
        ytdlp_cmd = _require_cmd(args.ytdlp)
        ffmpeg_cmd = shutil.which(args.ffmpeg) if args.backend in ("auto", "ffmpeg") else None
        gst_cmd = shutil.which(args.gst) if args.backend in ("auto", "gst") else None

        if backend == "auto":
            if ffmpeg_cmd:
                backend = "ffmpeg"
            elif gst_cmd:
                backend = "gst"
            else:
                raise SystemExit("No extraction backend available. Install ffmpeg or gst-launch-1.0.")
        if backend == "ffmpeg":
            ffmpeg_cmd = _require_cmd(args.ffmpeg)
        if backend == "gst":
            gst_cmd = _require_cmd(args.gst)

    processed_ids = set()
    total = 0
    skipped = 0
    for item_id, video_url, _row in _iter_files_check_rows(
        csv_path,
        id_col=args.id_col,
        url_col=args.url_col,
        state_col=args.state_col,
        require_success=args.require_success,
    ):
        if args.dedupe_id and item_id in processed_ids:
            continue
        processed_ids.add(item_id)
        total += 1

        id_dir = target_dir / item_id
        frames_dir = id_dir / "frames"
        if args.skip_existing and frames_dir.exists() and any(frames_dir.glob("frame_*.png")):
            skipped += 1
            continue

        if args.dry_run:
            continue

        id_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_with_ytdlp(
            video_url,
            id_dir,
            ytdlp_cmd,  # type: ignore[arg-type]
            outtmpl="video.%(ext)s",
        )

        if args.clean_frames:
            clean_existing_frames(frames_dir)

        _extract_frames(
            video_path,
            frames_dir,
            backend=backend,
            ffmpeg_cmd=ffmpeg_cmd or args.ffmpeg,
            gst_launch_cmd=gst_cmd or args.gst,
        )

    print(f"Done. ids={len(processed_ids)} total_rows_seen~={total} skipped_existing={skipped}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Download a video via yt-dlp and extract 1fps frames as frame_00000.png, frame_00001.png, ...\n"
            "Batch mode: read files_check.csv, download from video_url, and write to <target_dir>/<id>/frames/."
        ),
    )
    p.add_argument("url", nargs="?", default=None, help="Single video URL (or local filepath). Omit when using --csv.")
    p.add_argument("--csv", default=None, help="Batch input CSV (e.g. files_check.csv)")
    p.add_argument("--target-dir", default=".", help="Batch output root: creates <target_dir>/<id>/frames/")
    p.add_argument("--id-col", default="id", help="CSV column name for id (default: id)")
    p.add_argument("--url-col", default="video_url", help="CSV column name for video url (default: video_url)")
    p.add_argument("--state-col", default="state", help="CSV state column (default: state)")
    p.add_argument(
        "--require-success",
        action="store_true",
        help="Only process rows where state == 'Success' (case-insensitive).",
    )
    p.add_argument(
        "--dedupe-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process each id only once (default: True).",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory to download the video into (frames/ will be created next to it)",
    )
    p.add_argument("--ytdlp", default="yt-dlp", help="yt-dlp command (default: yt-dlp)")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg command (default: ffmpeg)")
    p.add_argument("--gst", default="gst-launch-1.0", help="gstreamer command (default: gst-launch-1.0)")
    p.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "gst"],
        default="auto",
        help="Frame extraction backend (default: auto prefers ffmpeg if available).",
    )
    p.add_argument(
        "--clean-frames",
        action="store_true",
        help="Remove existing frame_*.png in frames/ before extracting",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip ids that already have frame_*.png in frames/")
    p.add_argument("--dry-run", action="store_true", help="Parse CSV and plan work without downloading/extracting")
    args = p.parse_args()

    if args.csv:
        batch_from_files_check(args)
        return

    if not args.url:
        raise SystemExit("Provide a URL/path or use --csv for batch mode.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    maybe_local = Path(args.url).expanduser()
    if maybe_local.exists() and maybe_local.is_file():
        video_path = maybe_local.resolve()
        ytdlp_cmd = None
    else:
        ytdlp_cmd = _require_cmd(args.ytdlp)
        video_path = download_with_ytdlp(args.url, output_dir, ytdlp_cmd)

    frames_dir = video_path.parent / "frames"
    if args.clean_frames:
        removed = clean_existing_frames(frames_dir)
        if removed:
            print(f"Removed {removed} existing frames from {frames_dir}", file=sys.stderr)

    if args.backend == "ffmpeg":
        ffmpeg_cmd = _require_cmd(args.ffmpeg)
        extract_frames_1fps_ffmpeg(video_path, frames_dir, ffmpeg_cmd)
    elif args.backend == "gst":
        gst_cmd = _require_cmd(args.gst)
        extract_frames_1fps_gst(video_path, frames_dir, gst_cmd)
    else:
        ffmpeg_cmd = shutil.which(args.ffmpeg)
        if ffmpeg_cmd:
            extract_frames_1fps_ffmpeg(video_path, frames_dir, ffmpeg_cmd)
        else:
            gst_cmd = _require_cmd(args.gst)
            extract_frames_1fps_gst(video_path, frames_dir, gst_cmd)
    print(str(video_path))
    print(str(frames_dir))


if __name__ == "__main__":
    main()
