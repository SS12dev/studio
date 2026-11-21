# Setup checklist (phase 0)

Do these in order. Each one is independent of the code work happening in parallel.

1. **Clone the repo** into something like `~/dev/studio`, outside Desktop/Documents so iCloud does not sync model files.
2. **Install tools:** `xcode-select --install`, then `brew install uv ffmpeg`.
3. **Install and check:** `uv sync --python 3.12 --extra faces --extra raw --extra mac --extra dev`, then `uv run studio-check` and `uv run pytest -q`. Paste the `studio-check` output into the project thread.
4. **Free disk space:** aim for 150GB free. `df -h ~` shows it.
5. **Accounts (free):**
   - Hugging Face account + a read token (Settings > Access Tokens). Run `uv run huggingface-cli login`.
   - Request access on Hugging Face to `facebook/sam-3d-body-dinov3` (gated).
   - Register at https://smpl-x.is.tue.mpg.de to download SMPL-X (needed in phase 1).
   - Kaggle account with phone verification (unlocks ~30 free GPU hours a week).
   - Google account for Colab (free T4).
6. **Face sorting test set:** copy 100 to 300 photos into `~/Pictures/studio-test`: mixed solo and group shots of you and 3 to 5 friends, different years, lighting and angles. Include a few hard ones (sunglasses, side profile, small faces in crowds).
7. **Run the face bake-off** (see README), name the clusters, answer the review queue, and paste `studio-faces stats` output into the thread.
8. **LoRA set for yourself:** 20 to 40 photos of you following the checklist in the roadmap (close-up, half-body, full-body, several outfits and lighting). Keep them in `~/Pictures/studio-subjects/<name>/`.
9. **Consent:** before collecting a friend's photos for training, get a simple yes in writing (a chat message is fine) and note what they agreed to.
