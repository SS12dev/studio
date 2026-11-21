# Studio

A local, photo-first app for pulling a person and a pose out of photos, posing them in 3D,
and rendering the real person into a new pose. Runs on Apple Silicon (built for an M5 with 24GB).

Roadmap: https://claude.ai/artifact/XGwTadi1Jh8qRhXy44Ufho

## Status: phase 0

| Piece | State |
| --- | --- |
| Ingest (JPEG, PNG, TIFF, WebP, HEIC, RAW) | done |
| Memory-budgeted model manager (LRU, pinning) | done |
| Environment check (`studio-check`) | done |
| Face sorting bake-off CLI (`studio-faces`) | done, needs a run on real photos |
| FastAPI skeleton (`/health`, `/models`) | done |
| People page (`studio-faces ui`): fix groups, review queue, training exclusions | done |
| Frontend (React + React Flow + three.js) | not started |
| Extraction bake-off (`studio-bakeoff`: matting, depth, albedo) | done, needs a run on the Mac |
| Pose, 3D body and generator bake-offs | not started |

## Setup on the Mac

```bash
xcode-select --install                     # once
brew install uv ffmpeg                     # once
git clone git@github.com:<you>/<repo>.git && cd <repo>
uv sync --python 3.12
uv run studio-check                        # paste the output into the project thread
uv run pytest -q
```

## Face sorting bake-off

```bash
# 1. Detect and embed every face (first run downloads ~330MB of InsightFace weights)
uv run studio-faces scan ~/Pictures/studio-test -w runs/faces1

# 2. Group unknown faces and open the contact sheet
uv run studio-faces cluster -w runs/faces1
open runs/faces1/report.html

# 3. Name the groups you recognise (comma-separate groups that are the same person)
uv run studio-faces name "Person 1" "Shivam" -w runs/faces1
uv run studio-faces name "Person 4,Person 9" "Alex" -w runs/faces1

# 4. Scan a second folder into the same workspace (only new photos are processed),
#    then file them: confident ones go straight in, uncertain ones land in "Needs your answer"
uv run studio-faces scan ~/Pictures/studio-test-2 -w runs/faces1
uv run studio-faces assign -w runs/faces1
uv run studio-faces confirm "/path/photo.jpg#0" "Shivam" -w runs/faces1   # or "-" for nobody

# 5. Fix groups by hand in the browser: remove faces that are not the person, move faces
#    between people, mark filtered/edited photos "exclude from training", answer the review
#    queue with a click. Removals stick: assign never files that face under that person again.
uv run studio-faces ui -w runs/faces1

# 6. Timings, memory, accuracy and (after 50 answers) the calibrated threshold
uv run studio-faces stats -w runs/faces1
```

`assign` also runs a second pass over faces the quality gate set aside (small, blurry, turned
away). Those are matched only against people you already named, with thresholds 0.10 stricter,
and never start new groups. Turn it off with `--no-set-aside`.

Grouping uses average-linkage clustering at a 0.25 average cosine similarity. If two different
people end up in one group, re-run `cluster --link-sim 0.3`; if one person is split across groups,
try `--link-sim 0.2`, or just give both groups the same name to merge them.

For a stronger (slower) recogniser, scan into a fresh workspace with `--model antelopev2`.
Workspaces are tied to the model they were scanned with.

Routing defaults: auto-file when the best match scores at least 0.55 cosine **and** beats the
runner-up by 0.10; ask when it is in between; treat as someone new below 0.35. After 50 answers a
logistic calibrator replaces the fixed thresholds and auto-files only at 99% estimated precision.

Photos, embeddings and crops never go into git (see `.gitignore`).

## Extraction bake-off

```bash
make setup                                   # installs every dependency group
uv run studio-bakeoff list
uv run studio-bakeoff run ~/Pictures/studio-test -o runs/bake1 --limit 12
open runs/bake1/compare.html
```

Models run one at a time. First run downloads ~3GB of weights from Hugging Face.
`results.json` has load time, warm-up, median seconds per image and peak memory per model.

## Layout

```
backend/studio/
  api/            FastAPI app
  faces/          detect, quality gate, embed, cluster, assign, report, CLI
  models/         model registry + memory-budgeted manager
  ingest.py       any-format image decode
  envcheck.py     what this Mac can run
frontend/         (phase 1)
tests/            pytest, no models needed
docs/setup.md     step-by-step checklist
```

## Licences to know

InsightFace model weights and SMPL-X are non-commercial research licences. That is fine for this
personal project but would block selling it.
