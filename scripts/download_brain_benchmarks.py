"""Acquire unchanged author releases for IXI/TransMorph and LPBA40/SITReg.

Downloads and extraction only: no preprocessing, registration, or model imports.
Source archives, extracted files, and acquisition records stay under ignored Data/.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.parse
import urllib.request
from zipfile import ZipFile


SOURCES = {
    "ixi": [("IXI_data.zip", "https://drive.usercontent.google.com/download?id=1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol&export=download", "raw")],
    "lpba40": [
        ("LPBA40_Subjects_Delineation_Space_MRI_and_label_files.zip", "https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Delineation_Space_MRI_and_label_files.zip", "raw/delineation_package"),
        ("LPBA40_Subjects_Native_Space_MRI_data_and_masks.zip", "https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Native_Space_MRI_data_and_masks.zip", "raw/native_package"),
    ],
}
DIRECTORIES = {"ixi": "IXI_TransMorph", "lpba40": "LPBA40"}


def emit(**values):
    print(json.dumps(values), flush=True)


def save_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.active = True
            self.action = attrs["action"]
        if self.active and tag == "input" and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def open_download(url):
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    for _ in range(3):
        response = opener.open(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60,
        )
        if "text/html" not in response.headers.get("Content-Type", ""):
            return response
        page = response.read(1_000_000).decode("utf-8", errors="replace")
        response.close()
        form = DownloadForm()
        form.feed(page)
        if not form.action or urllib.parse.urlsplit(form.action).hostname != "drive.usercontent.google.com":
            raise RuntimeError("Download returned HTML instead of an archive; no supported public Drive confirmation form")
        url = form.action + "?" + urllib.parse.urlencode(form.fields)
    raise RuntimeError("Public Drive confirmation did not yield an archive")


def download(url, target):
    if target.exists() or target.with_suffix(target.suffix + ".partial").exists():
        raise FileExistsError(f"Refusing to overwrite an existing download: {target}")
    temporary = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    last_log = time.monotonic()
    with open_download(url) as response, temporary.open("xb") as output:
        expected = response.headers.get("Content-Length")
        expected = int(expected) if expected is not None else None
        while chunk := response.read(8 * 1024 * 1024):
            if count == 0 and not chunk.startswith(b"PK"):
                raise ValueError("Expected a ZIP archive")
            output.write(chunk)
            digest.update(chunk)
            count += len(chunk)
            if time.monotonic() - last_log >= 10:
                emit(downloading=target.name, bytes=count, expected_bytes=expected)
                last_log = time.monotonic()
    if not count or expected is not None and count != expected:
        raise ValueError(f"Incomplete archive: {target}")
    temporary.rename(target)
    return {"path": str(target), "source_url": url, "bytes": count, "sha256": digest.hexdigest()}


def extract_archive(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an extracted release: {destination}")
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        with ZipFile(archive) as source:
            entries = source.infolist()
            seen = set()
            for entry in entries:
                relative = PurePosixPath(entry.filename)
                if (relative.is_absolute() or ".." in relative.parts
                        or "\\" in entry.filename or not relative.parts
                        or ":" in relative.parts[0]
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError(f"Unsafe archive member: {entry.filename}")
                if relative in seen:
                    raise ValueError(f"Duplicate archive member: {entry.filename}")
                seen.add(relative)
            expanded = sum(entry.file_size for entry in entries)
            if expanded > shutil.disk_usage(temporary).free:
                raise OSError("Insufficient space for extracted release")
            # ZipFile checks each member's CRC while extracting it.
            source.extractall(temporary)
            members = [{"path": e.filename, "bytes": e.file_size, "crc32": f"{e.CRC:08x}"}
                       for e in entries if not e.is_dir()]
        temporary.rename(destination)
        return members
    except BaseException:
        shutil.rmtree(temporary)
        raise


def acquire_one(dataset, source, data_root):
    filename, url, relative_destination = source
    root = data_root / DIRECTORIES[dataset]
    archive = root / "source" / filename
    archive.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    emit(started=filename)
    record = download(url, archive)
    destination = root / relative_destination
    record["extracted_to"] = str(destination)
    record["members"] = extract_archive(archive, destination)
    record["nested_archives"] = []
    # LONI distributes each subject collection as a ZIP inside the release ZIP.
    for name in ("LPBA40subjects.delineation_space.zip", "LPBA40subjects.native_space.zip"):
        matches = list(destination.rglob(name))
        for nested in matches:
            nested_destination = nested.with_suffix("")
            with nested.open("rb") as nested_file:
                nested_sha256 = hashlib.file_digest(nested_file, "sha256").hexdigest()
            record["nested_archives"].append({
                "path": str(nested),
                "sha256": nested_sha256,
                "extracted_to": str(nested_destination),
                "members": extract_archive(nested, nested_destination),
            })
    record["seconds"] = time.monotonic() - started
    save_json(root / (filename + ".acquisition.json"), record)
    emit(completed=filename, bytes=record["bytes"], sha256=record["sha256"], seconds=record["seconds"])
    return dataset, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *SOURCES], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    selected = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for dataset in selected:
        root = data_root / DIRECTORIES[dataset]
        if root.exists():
            raise FileExistsError(f"Use a fresh destination; preserving {root}")
    jobs = [(dataset, source, data_root) for dataset in selected for source in SOURCES[dataset]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(acquire_one, *job) for job in jobs]
        results = [future.result() for future in futures]
    for dataset in selected:
        save_json(data_root / DIRECTORIES[dataset] / "acquisition.json", {
            "dataset": dataset,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing": "None. Source archives and extracted release members are unchanged.",
            "registration_or_training_executed": False,
            "archives": [record for key, record in results if key == dataset],
        })


if __name__ == "__main__":
    main()
