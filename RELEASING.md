# Releasing talkpipe-agents

There is no version string to bump. The package version is derived from the
git tag by `setuptools_scm`, so a release is a tag (and, for PyPI, a
published release) that triggers the publish jobs in CI.

## Tag conventions

Tags are PEP 440 versions with a `v` prefix, on `main`:

| Kind              | Tag          | Example      |
|-------------------|--------------|--------------|
| Final             | `vX.Y.Z`     | `v0.1.0`     |
| Alpha / beta      | `vX.Y.ZaN` / `vX.Y.ZbN` | `v0.2.0a1` |
| Release candidate | `vX.Y.ZrcN`  | `v0.2.0rc1`  |

## Steps

1. **Check the tree.** `main` is up to date, CI is green on it, and the
   working tree is clean.
2. **Test the release candidate.** Run the project test suite and exercise the
   example flows in the documentation, especially the packaged examples in
   the README and docs pages that show `llmAgent` in use. In practice that is
   `uv run pytest` plus a quick smoke-test of the README/docs snippets with a
   local provider or the scripted model path when no live model server is
   available. Do not tag a release until the library and its examples agree
   with the committed docs.
3. **TalkPipe floor.** The `talkpipe>=…` floor in `pyproject.toml` names the
   lowest TalkPipe release this version works with; raise it (and `uv lock`)
   first if this release relies on newer library API.
4. **Changelog.** Rename the `## Unreleased` section in `CHANGELOG.md` to
   `## X.Y.Z (YYYY-MM-DD)` and add a fresh empty `## Unreleased` above it.
   Land that through the normal branch → merge request flow.
5. **Tag** the merge commit on `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   Then advance `stable`, the repository's default branch, to the release.
   `stable` only ever points at a full release, so skip this for alphas,
   betas, and release candidates:

   ```bash
   git push origin vX.Y.Z^{commit}:stable
   ```

   Nothing copies commits between the two hosts, so push the same refs to
   GitHub as well (`github` here is a remote for
   `https://github.com/sandialabs/talkpipe-agents`; see [GitHub
   setup](#github-setup-once) for the first push):

   ```bash
   git push github main vX.Y.Z
   git push github vX.Y.Z^{commit}:stable   # full releases only, as above
   ```

6. **Publish.** CI builds and uploads; there are two publish jobs in
   `.github/workflows/ci-cd.yml`, one per hosting side, and each runs only
   after the full gate (tests, lint, package build and smoke test, security
   scan) has passed for the tagged commit.

   - **Self-hosted server (package registry).** Pushing the tag is enough:
     `publish-registry` uploads the sdist and wheel to the server's PyPI
     package registry (`<server>/api/packages/<owner>/pypi`), from where
     `pip install talkpipe-agents==X.Y.Z --index-url <server>/api/packages/<owner>/pypi/simple`
     fetches them on the same network. It authenticates with the
     `PUBLISH_TOKEN` Actions secret (an access token with package write;
     inherited from the owner) and fails loudly when the secret is missing
     or the built version is not the tag's. To turn this off once consumers
     install from PyPI, set the repository Actions variable
     `PUBLISH_REGISTRY` to `false`; the job is then skipped and nothing
     else changes.
   - **GitHub (PyPI).** Pushing the tag runs the gate on GitHub too but
     publishes nothing; the release does. On
     [GitHub](https://github.com/sandialabs/talkpipe-agents) *Releases →
     Draft a new release* for the tag, marking alphas, betas and release
     candidates as pre-releases. The `release: published` trigger runs
     `publish-package`, which builds the sdist and wheel, `twine check`s
     them, and uploads to PyPI with the `PYPI_API_TOKEN` repository secret.
     It ignores any tag that does not start with `v`, so a stray publish
     cannot upload to PyPI, and it never runs on the self-hosted server.
     Watch the run finish. **Only the GitHub release publishes to PyPI**: a
     release created on the self-hosted server is a record only.

   Both jobs may be live at once: the registry gets the tag push, PyPI gets
   the release.

7. **Verify** from a fresh environment that `pip install talkpipe-agents==X.Y.Z`
   (against the index you published to) resolves and
   `python -c "import talkpipe_agents; print(talkpipe_agents.__version__)"`
   prints `X.Y.Z`.

Nothing is bumped afterwards; the next commit on `main` reports itself as
`X.Y.(Z+1).devN` automatically.

## GitHub setup, once

The public repository is `https://github.com/sandialabs/talkpipe-agents`.
To set it up:

1. **Create the repository empty** — no README, license or `.gitignore`
   seeded, so the first push is a fast-forward.
2. **Push the history** from a clone of the self-hosted repository, then
   make `stable` the default branch (*Settings → General → Default branch*).
   GitHub defaults to `main` otherwise, and visitors would land on
   unreleased work:

   ```bash
   git remote add github git@github.com:sandialabs/talkpipe-agents.git
   git push github main stable --tags
   ```

3. **Secrets** (*Settings → Secrets and variables → Actions*):
   - `PYPI_API_TOKEN` — the token `publish-package` uploads with. The
     project does not exist on PyPI until the first upload, and PyPI only
     scopes tokens to existing projects, so this first token is
     account-wide; replace it with one scoped to `talkpipe-agents` once the
     first release is up.
   - `SAFETY_API_KEY` — optional. Present, `security-scan` uses Safety's
     commercial database gated by `.safety-policy.yml`; absent, it falls back
     to the free `safety check` database and still fails on a known
     vulnerability.
4. **Code scanning**: `codeql-analysis` runs from the workflow (*advanced
   setup*). Leave CodeQL's *default setup* off — GitHub rejects workflow
   uploads while it is on.

Nothing in the workflow refers to the self-hosted server by name; the jobs
that need github.com (`codeql-analysis`, `publish-package`, the Codecov and
artifact uploads) are gated on `github.server_url` and the jobs that need
the self-hosted server (`publish-registry`) on its opposite, so the same
file runs on both hosts.
