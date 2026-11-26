"""Capability prompt modules for the direct dev loop and job escalation (ADR-0018).

Each block is appended by `prompts.tool_system_prompt` only when the matching
tools are present in the turn. The wording is capability-generic (Rule H5): no
app, product, command or repository names.
"""

from collections.abc import Collection

DIRECT_DEV_TOOLS: frozenset[str] = frozenset(
    {
        "fs__list_files",
        "fs__read_file",
        "fs__grep",
        "fs__write_file",
        "fs__patch_file",
        "exec__run",
        "exec__status",
        "exec__write_stdin",
        "exec__cancel",
    }
)
AUTONOMOUS_JOB_FUNCTION = "autonomous_job"

_DIRECT_DEV_MODULE_VI = (
    "\n- Vòng làm việc trực tiếp trên mã nguồn (fs__* / exec__*):\n"
    "  + Đọc trước khi sửa: dùng `fs__grep` / `fs__read_file` với đường dẫn tuyệt đối, "
    "rồi sửa bằng `fs__patch_file` (đoạn `target` phải khớp đúng và duy nhất; lỗi "
    "`patch_target_ambiguous` nghĩa là cần đoạn dài hơn) thay vì ghi đè cả tệp.\n"
    "  + Kiểm chứng bằng `exec__run` với `cwd` của dự án. Nếu kết quả là `running` kèm "
    '`execTaskId`, gọi `job__wait` với `mode="exec_async"` và `execTaskId` đó; khi thức '
    "dậy, đọc `exec__status` để lấy kết quả thật. Chỉ báo thành công khi `status` là "
    "`completed` và đầu ra xác nhận yêu cầu.\n"
    "  + Khi lệnh phải tạo tệp, truyền `expectFiles` (đường dẫn tuyệt đối, tùy chọn "
    "`minBytes`/`magic`/`sha256`); `expected_output_missing` hoặc "
    "`expected_output_mismatch` nghĩa là việc chưa xong dù mã thoát là 0.\n"
    "  + Máy chủ tự phân loại rủi ro: lệnh xóa, mạng, hệ thống cần chủ ký duyệt; lệnh bị "
    "từ chối (`exec_hard_denied`, `exec_env_dump_forbidden`, `fs_path_forbidden`) không "
    "được lách bằng cách viết lại lệnh khác.\n"
    "  + Trước `mcp__call`, xem tên công cụ và tham số thật qua `mcp__list_servers`. Chỉ "
    "chuyển việc cho tác tử trong IDE (`workspace__coding_agent`) khi chủ yêu cầu rõ hoặc "
    "khi không có `exec__run`."
)

_DIRECT_DEV_MODULE_EN = (
    "\n- Direct source-code loop (fs__* / exec__*):\n"
    "  + Read before editing: use `fs__grep` / `fs__read_file` with absolute paths, then "
    "edit with `fs__patch_file` (the `target` must match exactly once; "
    "`patch_target_ambiguous` means use a longer target) instead of rewriting whole files.\n"
    "  + Verify with `exec__run` in the project `cwd`. When it returns `running` with an "
    '`execTaskId`, call `job__wait` with `mode="exec_async"` and that `execTaskId`; on '
    "wake, read `exec__status` for the real result. Report success only when `status` "
    "is `completed` and the output confirms the requirement.\n"
    "  + When a command must produce files, pass `expectFiles` (absolute paths, optional "
    "`minBytes`/`magic`/`sha256`); `expected_output_missing` or "
    "`expected_output_mismatch` means the work is not done even with exit code 0.\n"
    "  + The host classifies risk: delete, network and system commands need the owner's "
    "signature; refused commands (`exec_hard_denied`, `exec_env_dump_forbidden`, "
    "`fs_path_forbidden`) must not be worked around by rephrasing them.\n"
    "  + Before `mcp__call`, read the real tool names and arguments from "
    "`mcp__list_servers`. Delegate to an IDE agent (`workspace__coding_agent`) only when "
    "the owner asks for it or `exec__run` is unavailable."
)

_JOB_ESCALATION_MODULE_VI = (
    "\n- Việc nhiều bước: khi chủ yêu cầu từ 2 kết quả nối tiếp trở lên, một bản build/"
    "kiểm thử dài, hoặc cần chờ theo dõi nền, hãy gọi `autonomous_job` với "
    '`action="start"` và các bước đánh số rõ ràng thay vì cố làm hết trong một lượt.'
)

_JOB_ESCALATION_MODULE_EN = (
    "\n- Multi-step work: when the owner asks for 2 or more sequential deliverables, a "
    "long build/test, or a background wait, call `autonomous_job` with "
    '`action="start"` and clearly numbered steps instead of doing it all in one turn.'
)


def dev_capability_modules(active_names: Collection[str], *, english: bool) -> list[str]:
    """Returns the capability blocks that apply to the turn's active function names."""
    names = set(active_names)
    blocks: list[str] = []
    if names & DIRECT_DEV_TOOLS:
        blocks.append(_DIRECT_DEV_MODULE_EN if english else _DIRECT_DEV_MODULE_VI)
    if AUTONOMOUS_JOB_FUNCTION in names:
        blocks.append(_JOB_ESCALATION_MODULE_EN if english else _JOB_ESCALATION_MODULE_VI)
    return blocks
