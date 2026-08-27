# Step experiment archive

This directory maps each Feishu experiment-planning step to its frozen protocol
and corresponding final report. It exists so the plan/report lineage is visible
without reconstructing it from separate repository directories.

## Original research context

- Original idea and fixed hypothesis: [`context/ORIGINAL_IDEA.md`](context/ORIGINAL_IDEA.md)
  ([Feishu XML](context/ORIGINAL_IDEA.xml))
- Top-level experiment plan: [`context/EXPERIMENT_PLAN.md`](context/EXPERIMENT_PLAN.md)
  ([Feishu XML](context/EXPERIMENT_PLAN.xml))
- Feishu tokens and revisions: [`context/SOURCE.json`](context/SOURCE.json)

These files are direct v2 readbacks rather than reconstructions from the later
reports; Markdown-only trailing padding is normalized. They provide the
hypothesis and staged plan under which Step 1, Step 2, and Step 3 were
interpreted.

## Step-level plans and reports

| Step | Feishu plan snapshot | Frozen readable plan | Readable report | Feishu publication source | Source receipt |
| --- | --- | --- | --- | --- | --- |
| Step 1 | [`step1/PLAN_FEISHU_SNAPSHOT.xml`](step1/PLAN_FEISHU_SNAPSHOT.xml) | [`step1/PLAN.md`](step1/PLAN.md) | [`step1/REPORT.md`](step1/REPORT.md) | [`step1/REPORT_FEISHU_SOURCE.xml`](step1/REPORT_FEISHU_SOURCE.xml) | [`step1/SOURCE.json`](step1/SOURCE.json) |
| Step 2 / Stage-2A | [`step2/PLAN_FEISHU_SNAPSHOT.xml`](step2/PLAN_FEISHU_SNAPSHOT.xml) | [`step2/PLAN.md`](step2/PLAN.md) | [`step2/REPORT.md`](step2/REPORT.md) | [`step2/REPORT_FEISHU_SOURCE.xml`](step2/REPORT_FEISHU_SOURCE.xml) | [`step2/SOURCE.json`](step2/SOURCE.json) |
| Step 3 / Stage-R | [`step3/PLAN_FEISHU_SNAPSHOT.xml`](step3/PLAN_FEISHU_SNAPSHOT.xml) | [`step3/PLAN.md`](step3/PLAN.md) | [`step3/REPORT.md`](step3/REPORT.md) | [`step3/REPORT_FEISHU_SOURCE.xml`](step3/REPORT_FEISHU_SOURCE.xml) | [`step3/SOURCE.json`](step3/SOURCE.json) |

`PLAN_FEISHU_SNAPSHOT.xml` is a v2 readback of the current Feishu plan.
`PLAN.md` is the stricter preregistered protocol actually used for execution.
`REPORT.md` is the canonical readable result. `REPORT_FEISHU_SOURCE.xml` is the
stable XML submitted to Feishu; the current cloud revision and node/document
tokens are recorded in `SOURCE.json`. Expiring signed image-download URLs from
the live report readback are intentionally not committed.

`MANIFEST.json` records the size and SHA-256 digest of every context and
step-level archived artifact.
Regenerate the archive from an authenticated workstation with:

```bash
python3 scripts/archive_feishu_steps.py
```
