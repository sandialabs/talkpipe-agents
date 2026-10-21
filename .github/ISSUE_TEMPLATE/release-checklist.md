---
name: Release Checklist
about: Steps for doing a new release (see RELEASING.md for detail)
title: Release vX.Y.Z
labels: ''
assignees: ''

---

Assumes the TalkPipe release this depends on is already out and working.

- [ ] `main` up to date, CI green, working tree clean
- [ ] `talkpipe>=…` floor in `pyproject.toml` matches what this release needs (`uv lock` if changed)
- [ ] CHANGELOG: `## Unreleased` → `## X.Y.Z (YYYY-MM-DD)`, fresh empty section above; merged via MR
- [ ] Build the wheel and install it into a **new** virtual environment; `chatterlang_script` runs an `llmAgent[...]` step against a real model (Ollama or another configured provider)
- [ ] Anything found: fix via MR, re-test
- [ ] Tag the merge commit on `main` (`git tag -a vX.Y.Z -m vX.Y.Z`) and push the tag; advance `stable` for a full release
- [ ] Push `main`, the tag, and `stable` to GitHub
- [ ] Publish the release on GitHub (pre-release for alphas/betas/rcs); watch `publish-package` finish
- [ ] Verify: `pip install talkpipe-agents==X.Y.Z` in a fresh venv, `python -c "import talkpipe_agents; print(talkpipe_agents.__version__)"` prints `X.Y.Z`
