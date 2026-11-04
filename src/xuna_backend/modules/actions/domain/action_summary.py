"""Deterministic Vietnamese & English summaries and gesture mapping for `actions` (UDC-4, UDC-7).

Extracted from `entities.py` to keep domain entities focused and modular.
"""

from urllib.parse import urlparse

from xuna_backend.modules.cognition.domain.types import Locale

_BOARD_KEY_LABELS_VI: dict[str, str] = {
    "play_pause": "Phát/Tạm dừng (▶︎)",
    "next": "Bài tiếp (⏭)",
    "previous": "Bài trước (⏮)",
    "volume_up": "Tăng âm lượng",
    "volume_down": "Giảm âm lượng",
    "mute": "Tắt tiếng",
    "stop": "Dừng phát",
}

_BOARD_KEY_LABELS_EN: dict[str, str] = {
    "play_pause": "Play/Pause (▶︎)",
    "next": "Next track (⏭)",
    "previous": "Previous track (⏮)",
    "volume_up": "Volume Up",
    "volume_down": "Volume Down",
    "mute": "Mute",
    "stop": "Stop",
}


class ActionSummary:
    """Deterministic Vietnamese & English summaries for approval cards and chat chips."""

    @staticmethod
    def format_pending(
        tool: str,
        arguments: dict[str, object],
        target_alias: str,
        locale: Locale = Locale.VI,
    ) -> str:
        if locale == Locale.EN:
            return ActionSummary._format_pending_en(tool, arguments, target_alias)
        return ActionSummary._format_pending_vi(tool, arguments, target_alias)

    @staticmethod
    def _format_pending_vi(
        tool: str,
        arguments: dict[str, object],
        target: str,
    ) -> str:
        match tool:
            case "board.media_key":
                key = str(arguments.get("key", "play_pause"))
                label = _BOARD_KEY_LABELS_VI.get(key, key)
                return f"XUNA sẽ bấm {label} trên {target}"[:240]
            case "media.play_pause":
                return f"Phát/tạm dừng nhạc trên {target}"[:240]
            case "media.next":
                return f"Chuyển bài tiếp theo trên {target}"[:240]
            case "media.previous":
                return f"Quay lại bài trước trên {target}"[:240]
            case "media.volume_set":
                level = arguments.get("level", "")
                return f"Chỉnh âm lượng {level}% trên {target}"[:240]
            case "media.now_playing":
                return f"Xem bài đang phát trên {target}"[:240]
            case "media.play_search":
                q = str(arguments.get("query", "nhạc"))[:40]
                return f"Tìm và phát '{q}' trên {target}"[:240]
            case "app.launch":
                app_id = str(arguments.get("id", arguments.get("app_id", "")))[:40]
                return f"Mở {app_id} trên {target}"[:240]
            case "app.open_scheme":
                uri = str(arguments.get("uri", "ứng dụng"))[:40]
                return f"Mở liên kết nhanh {uri} trên {target}"[:240]
            case "app.list_allowed":
                return f"Xem danh sách ứng dụng trên {target}"[:240]
            case "notifications.list":
                return f"Đọc thông báo đang hiển thị trên {target}"[:240]
            case "notifications.action":
                act = str(arguments.get("action", "open"))[:20]
                return f"Thao tác thông báo ({act}) trên {target}"[:240]
            case "clipboard.read":
                return f"Đọc nội dung clipboard trên {target}"[:240]
            case "ui.inspect_tree":
                return f"Đọc cấu trúc giao diện màn hình trên {target}"[:240]
            case "ui.set_text":
                return f"Nhập văn bản vào ô nhập liệu trên {target}"[:240]
            case "ui.tap_text":
                txt = str(arguments.get("text", "nút"))[:32]
                return f"Chạm vào '{txt}' trên màn hình {target}"[:240]
            case "ui.long_press":
                txt = str(arguments.get("text", "phần tử"))[:32]
                return f"Nhấn giữ '{txt}' trên màn hình {target}"[:240]
            case "ui.scroll":
                direction = str(arguments.get("direction", "down"))[:12]
                return f"Cuộn màn hình ({direction}) trên {target}"[:240]
            case "ui.swipe_drag":
                return f"Vuốt/kéo trên màn hình {target}"[:240]
            case "ui.notifications":
                return f"Mở bảng thông báo trên {target}"[:240]
            case "ui.quick_settings":
                return f"Mở bảng cài đặt nhanh trên {target}"[:240]
            case "url.open":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or raw_url[:40]
                return f"Mở trang {host} trên {target}"[:240]
            case "browser.current_url":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Đọc trang đang mở ({browser}) trên {target}"[:240]
            case "workflow.run":
                wf_id = str(arguments.get("id", ""))[:40]
                return f"Chạy quy trình {wf_id} trên {target}"[:240]
            case "workflow.list":
                return f"Xem danh sách quy trình trên {target}"[:240]
            case "host.status":
                return f"Kiểm tra trạng thái {target}"[:240]
            case "host.notify":
                return f"Hiện thông báo trên {target}"[:240]
            case "display.wake":
                return f"Bật sáng màn hình {target}"[:240]
            case "workspace.open":
                proj = str(arguments.get("project", "recent"))[:40]
                ide = str(arguments.get("ide", "auto"))[:24]
                return f"Mở dự án {proj} ({ide}) trên {target}"[:240]
            case "workspace.git_summary":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Kiểm tra trạng thái Git dự án {proj} trên {target}"[:240]
            case "workspace.coding_agent":
                proj = str(arguments.get("project", "recent"))[:40]
                agent = str(arguments.get("agent", "auto"))[:24]
                return f"Mở {agent} tại dự án {proj} và tiếp tục công việc trên {target}"[:240]
            case "selection.read":
                return f"Đọc đoạn văn bản đang chọn trên {target}"[:240]
            case "host.lock_screen":
                return f"Khóa màn hình {target}"[:240]
            case "workspace.watch_task":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Theo dõi tiến độ tác vụ tại dự án {proj} trên {target}"[:240]
            case "mcp.list_servers":
                return f"Xem danh sách MCP server trên {target}"[:240]
            case "mcp.call":
                srv = str(arguments.get("server", ""))[:32]
                t_name = str(arguments.get("tool", ""))[:40]
                return f"Gọi công cụ MCP {srv}.{t_name} trên {target}"[:240]
            case "window.active_context":
                return f"Đọc ngữ cảnh cửa sổ đang mở trên {target}"[:240]
            case "browser.read_page":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Tóm tắt trang web đang mở ({browser}) trên {target}"[:240]
            case "beam.transfer":
                fname = str(arguments.get("fileName", "tệp"))[:40]
                return f"Truyền tệp {fname} qua XUNA Beam tới {target}"[:240]
            case "workflow.batch_steps":
                raw_steps = arguments.get("steps")
                count = len(raw_steps) if isinstance(raw_steps, list) else 0
                return f"Chạy chuỗi {count} bước liên hoàn trên {target}"[:240]
            case "action.undo":
                return f"Hoàn tác thao tác vừa thực hiện trên {target}"[:240]
            case "tv.list_devices":
                return f"Quét danh sách Smart TV trong mạng LAN từ {target}"[:240]
            case "tv.control":
                act = str(arguments.get("action", "play_pause"))[:24]
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Điều khiển {tv_dev} ({act}) qua {target}"[:240]
            case "tv.launch_app":
                app_id = str(arguments.get("appId", "youtube"))[:24]
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Mở ứng dụng {app_id} trên {tv_dev} qua {target}"[:240]
            case "tv.cast_media":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or "media"
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Chiếu nội dung từ {host} lên {tv_dev} qua {target}"[:240]
            case "smarthome.list_entities":
                return f"Xem danh sách thiết bị nhà thông minh từ {target}"[:240]
            case "smarthome.control":
                ent = str(arguments.get("entityId", "thiết bị"))[:40]
                act = str(arguments.get("action", "toggle"))[:24]
                return f"Điều khiển {ent} ({act}) qua {target}"[:240]
            case "clipboard.write":
                raw_txt = str(arguments.get("text", ""))
                return f"Chép {len(raw_txt)} ký tự vào clipboard của {target}"[:240]
            case "meeting.status":
                return f"Kiểm tra trạng thái cuộc họp và mic trên {target}"[:240]
            case "meeting.control":
                act = str(arguments.get("action", "toggle_mic"))[:24]
                return f"Điều khiển mic cuộc họp ({act}) trên {target}"[:240]
            case "dev.ports":
                return f"Quét cổng dev localhost trên {target}"[:240]
            case "system.vitals":
                return f"Kiểm tra pin, RAM và ổ cứng trên {target}"[:240]
            case "workspace.capture_capsule":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Đóng băng ngữ cảnh dự án {proj} trên {target}"[:240]
            case "workspace.restore_capsule":
                cap = str(arguments.get("capsuleId", arguments.get("project", "recent")))[:40]
                return f"Khôi phục ngữ cảnh dự án {cap} trên {target}"[:240]
            case "workspace.list_capsules":
                return f"Xem danh sách gói ngữ cảnh dự án trên {target}"[:240]
            case "pocket.push_item":
                kind = str(arguments.get("kind", "snippet"))[:24]
                return f"Cất mục ({kind}) vào túi thông minh trên {target}"[:240]
            case "pocket.list_items":
                return f"Xem các mục trong túi thông minh trên {target}"[:240]
            case "pocket.transform_item":
                mode = str(arguments.get("mode", "summarize"))[:24]
                return f"Chuyển đổi mục trong túi thông minh ({mode}) trên {target}"[:240]
            case "watchdog.watch_process_or_port":
                tk = str(arguments.get("targetKind", "port"))[:16]
                return f"Đặt đội tuần tra ({tk}) trên {target}"[:240]
            case "watchdog.watch_resource_threshold":
                metric = str(arguments.get("metric", "battery_below"))[:24]
                return f"Đặt cảnh báo ngưỡng tài nguyên ({metric}) trên {target}"[:240]
            case "watchdog.list_and_clear":
                return f"Kiểm tra danh sách đội tuần tra trên {target}"[:240]
            case "coach.decompose_overwhelm":
                goal = str(arguments.get("goal", "công việc"))[:40]
                return f"Chia nhỏ mục tiêu '{goal}' thành bước 5 phút trên {target}"[:240]
            case "meeting.desk_whisper_digest":
                topic = str(arguments.get("topic", "cuộc họp"))[:40]
                return f"Tạo nhắc bài thầm lặng ({topic}) từ {target}"[:240]
            case "habit.propose_workflow":
                return f"Phân tích thói quen đề xuất quy trình trên {target}"[:240]
            case "forge.synthesize_skill":
                goal = str(arguments.get("goal", "tác vụ tự động"))[:40]
                return f"Tự rèn kỹ năng mới ({goal}) trên {target}"[:240]
            case "forge.run_skill":
                sid = str(arguments.get("skillId", "kỹ năng"))[:32]
                return f"Chạy kỹ năng tự rèn {sid} trên {target}"[:240]
            case "forge.list_and_manage":
                return f"Quản lý thư viện kỹ năng tự rèn trên {target}"[:240]
            case "teach.record_session":
                mode = str(arguments.get("mode", "start"))[:24]
                return f"Học từ thao tác mẫu ({mode}) trên {target}"[:240]
            case "forge.extract_invoices_to_sheet":
                folder = str(arguments.get("sourceFolder", "HoaDon"))[:32]
                return f"Bóc tách hóa đơn trong {folder} ra bảng kê trên {target}"[:240]
            case "skill.markdown_bridge":
                act = str(arguments.get("action", "export_skill_md"))[:24]
                return f"Đồng bộ SKILL.md ({act}) trên {target}"[:240]
            case "skill.record_correction":
                sid = str(arguments.get("skillId", "kỹ năng"))[:32]
                return f"Ghi nhớ quy tắc tự sửa lỗi cho {sid} trên {target}"[:240]
            case "skill.workshop_sync":
                act = str(arguments.get("action", "sync_to_disk"))[:24]
                return f"Đồng bộ Skill Workshop ({act}) trên {target}"[:240]
            case "skill.chain_and_test":
                chain_ids = arguments.get("chainSkillIds")
                step_count = len(chain_ids) if isinstance(chain_ids, list) else 2
                return (
                    f"Chạy chuỗi liên hoàn {step_count} kỹ năng kèm kiểm thử tự động trên {target}"
                )[:240]
            case "memory.markdown_vault":
                act = str(arguments.get("action", "pre_compaction_flush"))[:28]
                return f"Đồng bộ kho ký ức Markdown ({act}) trên {target}"[:240]
            case "browser.semantic_snapshot":
                title = str(arguments.get("pageTitle", "trang web"))[:36]
                return f"Chụp cây ngữ nghĩa ARIA ({title}) tiết kiệm 85% token trên {target}"[:240]
            case "subagent.spawn_and_poll":
                act = str(arguments.get("action", "spawn"))[:20]
                return f"Điều phối Sub-Agent chạy nền ({act}) trên {target}"[:240]
            case "schedule.cron_and_hooks":
                evt = str(arguments.get("eventOrCron", "on_morning_wake"))[:28]
                return f"Quản lý Hook tự trị theo sự kiện ({evt}) trên {target}"[:240]
            case "heartbeat.configure_and_tick":
                chk = str(arguments.get("checkName", "tuần tra"))[:32]
                return f"Chạy nhịp tuần tra 0-token ({chk}) trên {target}"[:240]
            case "canvas.present":
                act = str(arguments.get("action", "present_invoice_table"))[:32]
                return f"Mở bảng tương tác Live Canvas ({act}) trên {target}"[:240]
            case "pipeline.execute_universal":
                goal = str(arguments.get("goal", "pipeline đa năng"))[:40]
                return f"Chạy pipeline nguyên tử vạn năng ({goal}) trên {target}"[:240]
            case _:
                return f"Thực hiện {tool} trên {target}"[:240]

    @staticmethod
    def _format_pending_en(
        tool: str,
        arguments: dict[str, object],
        target: str,
    ) -> str:
        match tool:
            case "board.media_key":
                key = str(arguments.get("key", "play_pause"))
                label = _BOARD_KEY_LABELS_EN.get(key, key)
                return f"XUNA will press {label} on {target}"[:240]
            case "media.play_pause":
                return f"Play/pause media on {target}"[:240]
            case "media.next":
                return f"Skip to next track on {target}"[:240]
            case "media.previous":
                return f"Previous track on {target}"[:240]
            case "media.volume_set":
                level = arguments.get("level", "")
                return f"Set volume to {level}% on {target}"[:240]
            case "media.now_playing":
                return f"Check now playing on {target}"[:240]
            case "media.play_search":
                q = str(arguments.get("query", "music"))[:40]
                return f"Search and play '{q}' on {target}"[:240]
            case "app.launch":
                app_id = str(arguments.get("id", arguments.get("app_id", "")))[:40]
                return f"Open {app_id} on {target}"[:240]
            case "app.open_scheme":
                uri = str(arguments.get("uri", "app"))[:40]
                return f"Open deep link {uri} on {target}"[:240]
            case "app.list_allowed":
                return f"List allowed apps on {target}"[:240]
            case "notifications.list":
                return f"List active notifications on {target}"[:240]
            case "notifications.action":
                act = str(arguments.get("action", "open"))[:20]
                return f"Perform notification action ({act}) on {target}"[:240]
            case "clipboard.read":
                return f"Read clipboard on {target}"[:240]
            case "ui.inspect_tree":
                return f"Inspect active screen UI tree on {target}"[:240]
            case "ui.set_text":
                return f"Enter text into input field on {target}"[:240]
            case "ui.tap_text":
                txt = str(arguments.get("text", "button"))[:32]
                return f"Tap '{txt}' on {target}"[:240]
            case "ui.long_press":
                txt = str(arguments.get("text", "element"))[:32]
                return f"Long-press '{txt}' on {target}"[:240]
            case "ui.scroll":
                direction = str(arguments.get("direction", "down"))[:12]
                return f"Scroll ({direction}) on {target}"[:240]
            case "ui.swipe_drag":
                return f"Swipe/drag on {target}"[:240]
            case "ui.notifications":
                return f"Open notifications shade on {target}"[:240]
            case "ui.quick_settings":
                return f"Open quick settings on {target}"[:240]
            case "url.open":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or raw_url[:40]
                return f"Open {host} on {target}"[:240]
            case "browser.current_url":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Read active browser tab ({browser}) on {target}"[:240]
            case "workflow.run":
                wf_id = str(arguments.get("id", ""))[:40]
                return f"Run workflow {wf_id} on {target}"[:240]
            case "workflow.list":
                return f"List workflows on {target}"[:240]
            case "host.status":
                return f"Check status of {target}"[:240]
            case "host.notify":
                return f"Show notification on {target}"[:240]
            case "display.wake":
                return f"Wake display on {target}"[:240]
            case "workspace.open":
                proj = str(arguments.get("project", "recent"))[:40]
                ide = str(arguments.get("ide", "auto"))[:24]
                return f"Open workspace {proj} ({ide}) on {target}"[:240]
            case "workspace.git_summary":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Read Git summary for {proj} on {target}"[:240]
            case "workspace.coding_agent":
                proj = str(arguments.get("project", "recent"))[:40]
                agent = str(arguments.get("agent", "auto"))[:24]
                return f"Launch {agent} in {proj} and continue work on {target}"[:240]
            case "selection.read":
                return f"Read selected text on {target}"[:240]
            case "host.lock_screen":
                return f"Lock screen on {target}"[:240]
            case "workspace.watch_task":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Watch task status in {proj} on {target}"[:240]
            case "mcp.list_servers":
                return f"List local MCP servers on {target}"[:240]
            case "mcp.call":
                srv = str(arguments.get("server", ""))[:32]
                t_name = str(arguments.get("tool", ""))[:40]
                return f"Call MCP tool {srv}.{t_name} on {target}"[:240]
            case "window.active_context":
                return f"Read active window context on {target}"[:240]
            case "browser.read_page":
                browser = str(arguments.get("browser", "auto"))[:20]
                return f"Summarize active web page ({browser}) on {target}"[:240]
            case "beam.transfer":
                fname = str(arguments.get("fileName", "file"))[:40]
                return f"Beam {fname} to {target}"[:240]
            case "workflow.batch_steps":
                raw_steps = arguments.get("steps")
                count = len(raw_steps) if isinstance(raw_steps, list) else 0
                return f"Run {count}-step batch workflow on {target}"[:240]
            case "action.undo":
                return f"Undo last action on {target}"[:240]
            case "tv.list_devices":
                return f"Scan Smart TVs on LAN from {target}"[:240]
            case "tv.control":
                act = str(arguments.get("action", "play_pause"))[:24]
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Control {tv_dev} ({act}) via {target}"[:240]
            case "tv.launch_app":
                app_id = str(arguments.get("appId", "youtube"))[:24]
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Launch {app_id} on {tv_dev} via {target}"[:240]
            case "tv.cast_media":
                raw_url = str(arguments.get("url", ""))
                host = urlparse(raw_url).hostname or "media"
                tv_dev = str(arguments.get("device", "TV"))[:32]
                return f"Cast {host} to {tv_dev} via {target}"[:240]
            case "smarthome.list_entities":
                return f"List smart-home entities via {target}"[:240]
            case "smarthome.control":
                ent = str(arguments.get("entityId", "device"))[:40]
                act = str(arguments.get("action", "toggle"))[:24]
                return f"Control {ent} ({act}) via {target}"[:240]
            case "clipboard.write":
                raw_txt = str(arguments.get("text", ""))
                return f"Copy {len(raw_txt)} characters to clipboard on {target}"[:240]
            case "meeting.status":
                return f"Check meeting & microphone status on {target}"[:240]
            case "meeting.control":
                act = str(arguments.get("action", "toggle_mic"))[:24]
                return f"Control meeting microphone ({act}) on {target}"[:240]
            case "dev.ports":
                return f"Scan active localhost dev ports on {target}"[:240]
            case "system.vitals":
                return f"Check battery, memory, and disk vitals on {target}"[:240]
            case "workspace.capture_capsule":
                proj = str(arguments.get("project", "recent"))[:40]
                return f"Capture workspace capsule for {proj} on {target}"[:240]
            case "workspace.restore_capsule":
                cap = str(arguments.get("capsuleId", arguments.get("project", "recent")))[:40]
                return f"Restore workspace capsule {cap} on {target}"[:240]
            case "workspace.list_capsules":
                return f"List workspace capsules on {target}"[:240]
            case "pocket.push_item":
                kind = str(arguments.get("kind", "snippet"))[:24]
                return f"Push item ({kind}) into universal pocket on {target}"[:240]
            case "pocket.list_items":
                return f"List universal pocket items on {target}"[:240]
            case "pocket.transform_item":
                mode = str(arguments.get("mode", "summarize"))[:24]
                return f"Transform pocket item ({mode}) on {target}"[:240]
            case "watchdog.watch_process_or_port":
                tk = str(arguments.get("targetKind", "port"))[:16]
                return f"Register background watchdog ({tk}) on {target}"[:240]
            case "watchdog.watch_resource_threshold":
                metric = str(arguments.get("metric", "battery_below"))[:24]
                return f"Register resource threshold watchdog ({metric}) on {target}"[:240]
            case "watchdog.list_and_clear":
                return f"List or clear background watchdogs on {target}"[:240]
            case "coach.decompose_overwhelm":
                goal = str(arguments.get("goal", "task"))[:40]
                return f"Decompose '{goal}' into a 5-minute starter step on {target}"[:240]
            case "meeting.desk_whisper_digest":
                topic = str(arguments.get("topic", "meeting"))[:40]
                return f"Prepare silent desk whisper ({topic}) on {target}"[:240]
            case "habit.propose_workflow":
                return f"Mine repeated habits to propose workflow on {target}"[:240]
            case "forge.synthesize_skill":
                goal = str(arguments.get("goal", "automation"))[:40]
                return f"Synthesize personal skill ({goal}) on {target}"[:240]
            case "forge.run_skill":
                sid = str(arguments.get("skillId", "skill"))[:32]
                return f"Run forged skill {sid} on {target}"[:240]
            case "forge.list_and_manage":
                return f"Manage forged skill library on {target}"[:240]
            case "teach.record_session":
                mode = str(arguments.get("mode", "start"))[:24]
                return f"Record Show-Me-Once demonstration ({mode}) on {target}"[:240]
            case "forge.extract_invoices_to_sheet":
                folder = str(arguments.get("sourceFolder", "Invoices"))[:32]
                return f"Extract invoices in {folder} to spreadsheet on {target}"[:240]
            case "skill.markdown_bridge":
                act = str(arguments.get("action", "export_skill_md"))[:24]
                return f"Bridge SKILL.md ({act}) on {target}"[:240]
            case "skill.record_correction":
                sid = str(arguments.get("skillId", "skill"))[:32]
                return f"Record self-healing correction for {sid} on {target}"[:240]
            case "skill.workshop_sync":
                act = str(arguments.get("action", "sync_to_disk"))[:24]
                return f"Sync Skill Workshop ({act}) on {target}"[:240]
            case "skill.chain_and_test":
                chain_ids = arguments.get("chainSkillIds")
                step_count = len(chain_ids) if isinstance(chain_ids, list) else 2
                return f"Run {step_count}-step skill chain with self-test on {target}"[:240]
            case "memory.markdown_vault":
                act = str(arguments.get("action", "pre_compaction_flush"))[:28]
                return f"Sync transparent Markdown memory vault ({act}) on {target}"[:240]
            case "browser.semantic_snapshot":
                title = str(arguments.get("pageTitle", "web page"))[:36]
                return f"Capture ARIA semantic snapshot ({title}) on {target}"[:240]
            case "subagent.spawn_and_poll":
                act = str(arguments.get("action", "spawn"))[:20]
                return f"Orchestrate background sub-agent ({act}) on {target}"[:240]
            case "schedule.cron_and_hooks":
                evt = str(arguments.get("eventOrCron", "on_morning_wake"))[:28]
                return f"Manage autonomous cron/lifecycle hook ({evt}) on {target}"[:240]
            case "heartbeat.configure_and_tick":
                chk = str(arguments.get("checkName", "patrol"))[:32]
                return f"Run zero-token heartbeat ({chk}) on {target}"[:240]
            case "canvas.present":
                act = str(arguments.get("action", "present_invoice_table"))[:32]
                return f"Present Live Canvas A2UI ({act}) on {target}"[:240]
            case "pipeline.execute_universal":
                goal = str(arguments.get("goal", "universal pipeline"))[:40]
                return f"Execute universal composable pipeline ({goal}) on {target}"[:240]
            case _:
                return f"Run {tool} on {target}"[:240]

    @staticmethod
    def format_result(
        tool: str,
        arguments: dict[str, object],
        target_alias: str,
        status: str,
        locale: Locale = Locale.VI,
    ) -> str:
        base = ActionSummary.format_pending(tool, arguments, target_alias, locale)
        if locale == Locale.EN:
            if status == "ok":
                return f"✓ {base}"[:240]
            return f"✗ {base} ({status})"[:240]
        if status == "ok":
            return f"✓ Đã thực hiện: {base}"[:240]
        reason_map = {
            "needs_permission": "thiếu quyền hệ thống",
            "denied": "bị từ chối bởi chính sách",
            "rejected": "bạn đã từ chối",
            "expired": "hết thời gian xác nhận",
            "timeout": "quá thời gian phản hồi",
            "unsupported": "không hỗ trợ",
            "failed": "lỗi thực thi",
        }
        reason = reason_map.get(status, status)
        return f"✗ Không thực hiện được: {base} ({reason})"[:240]


def gesture_for_action_status(status: str) -> str | None:
    """Deterministic gesture mapping (UDC-4 §5.4):
    - `ok` -> `bounce_play`
    - `failed`, `timeout`, `needs_permission`, `denied` -> `look_around`
    - `rejected` / `expired` -> None
    """
    if status == "ok":
        return "bounce_play"
    if status in ("failed", "timeout", "needs_permission", "denied"):
        return "look_around"
    return None
