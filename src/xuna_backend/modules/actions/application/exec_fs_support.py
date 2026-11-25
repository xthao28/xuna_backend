"""Backend rules for the Wave 3 `exec.*` / `fs.*` host tools (ADR-0018).

- `exec.run`, `exec.write_stdin`, `fs.write_file` and `fs.patch_file` must be
  classified by the host through `xuna/preflight` before every call. A host that
  cannot answer (it predates preflight, is not a desktop agent, or is the
  in-process self host) is refused with `preflight_required`; the backend never
  substitutes its own estimate and never lowers a risk the host reported.
- `exec_async` watches bind only to an explicit `execTaskId` returned by
  `exec.run` (Rule H3: no guessed target).
"""

import re

from xuna_backend.modules.actions.domain.watch import WatchTarget
from xuna_backend.modules.hosts.application.xtp_frames import HostPreflight

EXEC_FS_PREFLIGHT_TOOLS: frozenset[str] = frozenset(
    {"exec.run", "exec.write_stdin", "fs.write_file", "fs.patch_file"}
)
EXEC_TASK_ID_RE = re.compile(r"^exec_[a-zA-Z0-9_-]{1,64}$")


def requires_mandatory_preflight(tool_name: str) -> bool:
    """True for tools the backend may run only after a host preflight answer."""
    return tool_name in EXEC_FS_PREFLIGHT_TOOLS


def missing_preflight_refusal() -> HostPreflight:
    """Refusal used when a mandatory preflight could not be obtained."""
    return HostPreflight(
        status="denied",
        effective_risk=None,
        reason_code=None,
        error_code="preflight_required",
        element_role=None,
        element_name=None,
    )


def bind_exec_async_target(raw_task_id: object) -> tuple[WatchTarget | None, str | None]:
    """Binds an `exec_async` watch to the explicit task handle, or reports why not."""
    if raw_task_id is None:
        return None, "watch_unsupported"
    if not isinstance(raw_task_id, str) or not EXEC_TASK_ID_RE.fullmatch(raw_task_id):
        return None, "invalid_args"
    return WatchTarget(exec_task_id=raw_task_id), None


def exec_async_verify_instruction(task_id: str | None) -> str:
    """Wake-up guidance: the exit code must be read before anything is reported."""
    handle = f" (taskId={task_id})" if task_id else ""
    return (
        f"Call read-only exec__status{handle} first and report the real exit code and "
        "output (Rule H1); a settled watch only means the process ended, not that it "
        "succeeded. If it is still running, call job__wait again."
    )
