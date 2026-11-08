# Team handoff — November 8, 2026

## Open the shared controller from your own laptop

```sh
ssh -N -L 8081:127.0.0.1:8081 pah@10.103.0.132
```

Keep that terminal open, then open http://localhost:8081/. Each teammate needs their own tunnel; the old laptop tunnel disappears when that laptop shuts down. The controller itself runs on the NUC in tmux session `unified-arm-control`, independently of the laptop. Do not launch a second controller to view the page.

Station repository: `/home/pah/physical-ai-hacks`. Runtime config, credentials, calibration, recordings and saved flow settings stay on the NUC. The active service has the integrated orchestration, Star, model selection and sequence monitor controls. Publishing this handoff does not start automation or change operator inputs.

## What is published

- `main`: integrated central controller, fullscreen flow editor, editable 29-block competition strategy, Star controls, model adapter, latest sequence monitors, CV-aligned USB bundle pull and standalone sponsor bonus demo.
- `codex/laptop-handoff`: source snapshot of the laptop's unfinished project work.
- `codex/nuc-deployed-handoff`: exact source overlay used in the tested NUC deployment, including the recovered joint-model adapter.
- Original feature branches remain available: `codex/orchestration`, `codex/central-controller`, `codex/star-record-replay`, `codex/so101-dashboard`, `codex/cv-aligned-usb-bundle-pull`, `codex/local-sponsor-bonus-demo`.

## Continue development

Start from a fresh clone of remote `main`; the original laptop checkout has intentionally not been reset or cleaned. Existing controller state on the NUC is authoritative during the demo.

Read [controller operation](docs/unified_control.md), [orchestration extension points](docs/orchestration.md), [competition strategy and dummy bindings](docs/demo_strategy.md), [sequence monitors](docs/sequence_monitors.md), [USB pull replay](docs/usb_bundle_pull.md), and [sponsor bonus demo](bonus_demo/README.md).

Recorded episode paths and trained checkpoint IDs in the strategy are editable dummy bindings where real assets have not been recorded yet. Replace these in the operation blocks/bindings before executing the competition map. The VLM adapter/protocol is implemented; passing mocked adapter tests does not establish real camera verification accuracy.

## Laptop asset backup on the NUC

Backup root: `/home/pah/laptop-handoff-20261108/`.

- `downloads/`: local training exports, datasets and downloaded archives.
- `recordings/`: replay assets and small demo recordings (also available in source where applicable).
- `laptop-inspection-backup.tar.gz`: deployment scripts, QA artifacts and recovered pre/post-reboot source; operational archive, not a Git source directory.
- `team-handoff-source.tar.gz` and `integration/`: integrated source used for handoff validation.

Bulk datasets and operational archives are on the NUC rather than in GitHub. Local virtual environments and regenerable caches are excluded. Keep station credentials in the existing station configuration, never in Git.

## Validation

The combined source passed 153 tests plus 11 subtests on the NUC in an isolated directory, covering orchestration, strategy, VLM adapters, unified controller, sequence monitors, model selection/native joint checkpoints, Star recording/replay, USB replay, arm server and SO-101. No robot run was started for validation. `asset-verification.json` in the backup root records the dataset checksum verification result. The latest rubric is backed up there as `hillclimb_rubric_v4.pdf`.
