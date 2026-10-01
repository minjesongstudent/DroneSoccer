"""PreToolUse hook for the autopilot Claude: block every write outside the MDP.

Claude Code runs this before each Edit, MultiEdit, Write or NotebookEdit call
and passes the call as JSON on stdin. Exit 2 blocks the call, and stderr goes
back to Claude as the reason, so it can correct itself. Exit 0 lets it through.

Only two files may be written, both inside the sandbox $AUTOPILOT_WORK:
contract.py, and only if the result still passes mdp_guard.check against the
pristine copy $AUTOPILOT_WORK/../contract.before.py, and proposal.md.
The edit is applied in memory and checked before it reaches the disk.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdp_guard import check  # noqa: E402


def block(reason: str):
    print(f"BLOCKED by the autopilot guard: {reason}", file=sys.stderr)
    sys.exit(2)


def after_edit(text: str, old: str, new: str, replace_all: bool) -> str | None:
    """The file after one Edit, or None if the edit would not apply."""
    if old not in text or (not replace_all and text.count(old) != 1):
        return None
    return text.replace(old, new) if replace_all else text.replace(old, new, 1)


def main():
    work = os.environ.get("AUTOPILOT_WORK")
    if not work:
        block("AUTOPILOT_WORK is not set, so no write can be allowed")
    work = Path(work).resolve()
    call = json.load(sys.stdin)
    tool, args = call.get("tool_name", ""), call.get("tool_input", {})
    if tool not in ("Edit", "MultiEdit", "Write"):
        block(f"{tool} is not allowed; only Edit/Write of contract.py and proposal.md")

    path = Path(args.get("file_path", ""))
    if not path.is_absolute():
        path = Path(call.get("cwd", work)) / path
    path = path.resolve()   # follows symlinks and .., so no path trick lands elsewhere
    contract, proposal = work / "contract.py", work / "proposal.md"
    if path == proposal:
        sys.exit(0)
    if path != contract:
        block(f"{path} is outside the MDP. Only {contract} and {proposal} may be written. "
              "The simulator, environment and training code are frozen.")

    text = contract.read_text()
    if tool == "Write":
        text = args.get("content", "")
    else:
        edits = args.get("edits") if tool == "MultiEdit" else [args]
        for e in edits or []:
            text = after_edit(text, e.get("old_string", ""), e.get("new_string", ""),
                              bool(e.get("replace_all")))
            if text is None:
                sys.exit(0)   # the tool itself will reject an edit that does not apply
    problems = check((work.parent / "contract.before.py").read_text(), text)
    if problems:
        block("this change leaves the MDP surface:\n  " + "\n  ".join(problems))
    sys.exit(0)


if __name__ == "__main__":
    main()
