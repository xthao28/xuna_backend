"""Approval-card and chip text for the Wave 3 `exec.*` / `fs.*` tools (ADR-0018).

The card shows the real command, working directory and file path so the owner
signs exactly what will run; the approval itself stays bound to the full
arguments (`argsSha256`). Long values are cut with an ellipsis to fit the
240-character summary limit.
"""


def _cut(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def exec_fs_pending_text(
    tool: str, arguments: dict[str, object], target: str, *, english: bool
) -> str | None:
    """Returns the pending-card text for an `exec.*` / `fs.*` tool, else `None`."""
    path = _cut(arguments.get("path"), 90)
    match tool:
        case "exec.run":
            command = _cut(arguments.get("command"), 150)
            cwd = _cut(arguments.get("cwd") or "~", 50)
            if english:
                return f"Run `{command}` in {cwd} on {target}"[:240]
            return f"Chạy lệnh `{command}` trong {cwd} trên {target}"[:240]
        case "exec.write_stdin":
            text = _cut(arguments.get("input"), 120)
            task = _cut(arguments.get("taskId"), 40)
            if english:
                return f"Send input '{text}' to {task} on {target}"[:240]
            return f"Gửi '{text}' vào lệnh {task} trên {target}"[:240]
        case "exec.cancel":
            task = _cut(arguments.get("taskId"), 40)
            if english:
                return f"Stop command {task} on {target}"[:240]
            return f"Dừng lệnh {task} trên {target}"[:240]
        case "exec.status":
            task = _cut(arguments.get("taskId"), 40)
            if english:
                return f"Check command {task} on {target}"[:240]
            return f"Xem kết quả lệnh {task} trên {target}"[:240]
        case "fs.write_file":
            size = len(str(arguments.get("content") or ""))
            if english:
                return f"Write file {path} ({size} chars) on {target}"[:240]
            return f"Ghi tệp {path} ({size} ký tự) trên {target}"[:240]
        case "fs.patch_file":
            old = _cut(arguments.get("target"), 60)
            if english:
                return f"Edit file {path}: replace '{old}' on {target}"[:240]
            return f"Sửa tệp {path}: thay '{old}' trên {target}"[:240]
        case "fs.read_file" | "fs.grep" | "fs.list_files":
            if english:
                return f"Read {path} on {target}"[:240]
            return f"Đọc {path} trên {target}"[:240]
    return None
