# PAI run evidence

## Controller and source

- Canonical registry: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry`
- Registry HEAD: `1e70bd145c04f3c21f64b691232cefdc67d51846`
- Registry dirty snapshot count at submission: 212
- `bin/pai-job` SHA256:
  `682a9028ee2a889ad01ea3db823f26fe6477c7bbe4212823e0f0eb80a6c438ba`
- `config/resources.json` SHA256:
  `ab1e9ebf731e5278b6f591d8db3bdd95d8e04ea647363a879cf6e1cf447be91f`
- `config/toolchain.json` SHA256:
  `b8e03b0a520e99d01d7645c20865142cda30df27b80cd57e473635e9f417eb79`
- Pinned DLC SHA256:
  `09fac825e088dfeee7f55919d6ad8421d4f46c2a6554da1827b664a08518473c`
- Task source commit at r3 submission:
  `719e437fc4701e3b03e6de1dfb39c3293b7eb18a`
- Template SHA256:
  `eafe0d675dd14a89d77166d45e23b01724377501b847f56c1e4622ef8e2177b1`
- Launcher/payload SHA256:
  `3aaa45a315bab31c438a679d2eac6b099a04770644fcd9e55017be2c2a2f9918`
- Runtime manifest SHA256:
  `0a6103aa42202e50e4a6f10331a4f534293b2b2c52790799707f30596373adc7`

## Successful formal evaluation

- Run ID: `r142-fp11-stage1-eval-20260823-1412-r3`
- JobId: `dlc1e1wg0af86rlq`
- Status: `Succeeded`
- ResourceId: `quotaewyznuc7b9l`
- Exact idle evidence: `UseOversoldResource=true`
- OversoldType: `AcceptQuotaOverSold`
- Shape: 1x8 GPU, 92 CPU, 1600Gi memory and shared memory
- `DisableEcsStockCheck=true`
- AIMaster: Sync OnFailure, max 50 platform restarts
- Runtime identity and first artifacts: `2254:2254`
- W&B entity gate: exact `chen_jian-cj-workspace`, viewer `chen_jian`,
  admin, pending=false, identity inference disabled; no credentials recorded
- First work: persisted completed shard-07
- Completion: 8/8 shards, 400 paired episodes, 10 policies
- PAI probe created: no
- Browser used: no (`browser_not_used=fifo_not_applicable`)

## Repaired predecessor cleanup

- Predecessor run: `r142-fp11-stage1-eval-20260823-1323-r2`
- Predecessor JobId: `dlc1yavs3uab0cm0`
- Terminal state: `Stopped`, `ReasonCode=StoppedByUser`
- First work: absent
- Delete preflight: prepared=true
- Delete result: deleted=true, verified_absent=true
- Deleted object: exact PAI service row only
- Preserved: registry run, CPFS directories, placement evidence, payload,
  readbacks, stop/error report
- Succeeded or active jobs deleted: none
