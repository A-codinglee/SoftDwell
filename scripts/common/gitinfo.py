# scripts/common/gitinfo.py
"""
The code version a run was started with, for train_config.json:

    train_config["git"] = git_info()

-> {"commit", "branch", "dirty", "changed_files"} of the SoftDwell repository. Never raises: if git or the
repository is not available, the block says why and the run goes on.
"""
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*args):
    # rstrip only: porcelain status lines start with a space (" M path")
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=20,
                          check=True).stdout.rstrip()


def git_info(max_files: int = 50) -> dict:
    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        # tracked files with uncommitted changes, plus new files git would commit (ignored files excluded)
        changed = [ln[3:] for ln in _git("status", "--porcelain", "--untracked-files=all").splitlines() if ln]
        return {"commit": commit, "branch": branch, "dirty": bool(changed),
                "changed_files": changed[:max_files] + (["..."] if len(changed) > max_files else [])}
    except Exception as e:      # no git binary, not a repository, timeout, ...
        return {"commit": None, "error": f"{type(e).__name__}: {e}"[:300]}
