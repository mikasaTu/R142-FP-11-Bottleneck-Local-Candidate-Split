#!/usr/bin/env python3
"""Archive the Feishu step plans and pair them with repository reports.

The command reads document metadata/content through lark-cli v2. It stores the
current plan XML, the frozen readable protocol, the readable report, and the
XML source used to publish that report. Report readback is used to record the
current Feishu revision without persisting temporary signed image URLs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPO_ROOT / "docs" / "steps"

STEPS = {
    "step1": {
        "plan": {
            "wiki_node_token": "BWy7wpbVjieM4QknvdxcfGABnFc",
            "document_token": "J4I0d5mjKo41zWxlbq3c9uYjnYy",
            "wiki_url": "https://icnbwz7kd1ui.feishu.cn/wiki/BWy7wpbVjieM4QknvdxcfGABnFc",
        },
        "report": {
            "wiki_node_token": "WRkEwByXUiRnLrklo0VcPiApnKh",
            "document_token": "FZVsdynvHocFCAxydO9c3pHbnwe",
            "wiki_url": "https://icnbwz7kd1ui.feishu.cn/wiki/WRkEwByXUiRnLrklo0VcPiApnKh",
        },
        "canonical_plan": "docs/PREREGISTERED_PROTOCOL.md",
        "canonical_report": "reports/STAGE1_EXPERIMENT_REPORT.md",
        "report_feishu_source": "reports/FEISHU_EXPERIMENT_REPORT.xml",
    },
    "step2": {
        "plan": {
            "wiki_node_token": "RTaXwkdFuiEGoVkS9qQclkSHnxc",
            "document_token": "QzmHdVetdoNDdBxuJcCcOqM9nGb",
            "wiki_url": "https://icnbwz7kd1ui.feishu.cn/wiki/RTaXwkdFuiEGoVkS9qQclkSHnxc",
        },
        "report": {
            "wiki_node_token": "RQCRwCzFlist2hkY776cCYkqnVd",
            "document_token": "D5BtdfVyEozHydxEHmnc2e0Unjc",
            "wiki_url": "https://icnbwz7kd1ui.feishu.cn/wiki/RQCRwCzFlist2hkY776cCYkqnVd",
        },
        "canonical_plan": "docs/STAGE2A_PREREGISTERED_PROTOCOL.md",
        "canonical_report": "reports/STAGE2A_EXPERIMENT_REPORT.md",
        "report_feishu_source": "reports/FEISHU_STAGE2A_EXPERIMENT_REPORT.xml",
    },
}


def fetch_document(document_token: str) -> dict:
    completed = subprocess.run(
        [
            "lark-cli",
            "docs",
            "+fetch",
            "--doc",
            document_token,
            "--api-version",
            "v2",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not payload.get("ok"):
        raise RuntimeError(f"Feishu fetch failed for {document_token}: {payload}")
    return payload["data"]["document"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": fetched_at,
        "source": "Feishu docs v2 readback plus canonical repository artifacts",
        "files": [],
    }

    for step_name, spec in STEPS.items():
        target = ARCHIVE_ROOT / step_name
        target.mkdir(parents=True, exist_ok=True)

        plan_doc = fetch_document(spec["plan"]["document_token"])
        report_doc = fetch_document(spec["report"]["document_token"])

        write_text(target / "PLAN_FEISHU_SNAPSHOT.xml", plan_doc["content"] + "\n")
        shutil.copyfile(REPO_ROOT / spec["canonical_plan"], target / "PLAN.md")
        shutil.copyfile(REPO_ROOT / spec["canonical_report"], target / "REPORT.md")
        shutil.copyfile(
            REPO_ROOT / spec["report_feishu_source"],
            target / "REPORT_FEISHU_SOURCE.xml",
        )

        source = {
            "schema_version": 1,
            "fetched_at_utc": fetched_at,
            "step": step_name,
            "plan": {
                **spec["plan"],
                "revision_id": plan_doc["revision_id"],
                "snapshot_file": "PLAN_FEISHU_SNAPSHOT.xml",
            },
            "report": {
                **spec["report"],
                "revision_id": report_doc["revision_id"],
                "readable_file": "REPORT.md",
                "publication_source_file": "REPORT_FEISHU_SOURCE.xml",
                "note": (
                    "The live report was read back to verify its current revision. "
                    "REPORT_FEISHU_SOURCE.xml is the stable publication input and "
                    "omits expiring signed image-download URLs returned by readback."
                ),
            },
        }
        write_text(
            target / "SOURCE.json",
            json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

        for filename in (
            "PLAN.md",
            "PLAN_FEISHU_SNAPSHOT.xml",
            "REPORT.md",
            "REPORT_FEISHU_SOURCE.xml",
            "SOURCE.json",
        ):
            path = target / filename
            manifest["files"].append(
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    write_text(
        ARCHIVE_ROOT / "MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
