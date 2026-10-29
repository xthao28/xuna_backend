"""Curated Domain Packs for XUNA Knowledge Expert (plan §6.5, Appendix A, KX-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SafetyTier(StrEnum):
    GENERAL = "general"
    HIGH_STAKES = "high_stakes"


@dataclass(frozen=True, slots=True)
class DomainPack:
    pack_id: str  # "embedded", "software", "finance", "marketing", "law", "biomed", "language"
    name: str  # Tên gợi ý tiếng Việt
    description: str  # ≤ 240 ký tự: từ khóa để lexical/embedding match
    brief: str  # ≤ 1500 ký tự: phương pháp trả lời chuyên sâu
    safety_tier: SafetyTier


PACKS: tuple[DomainPack, ...] = (
    DomainPack(
        pack_id="embedded",
        name="Nhúng & phần cứng",
        description=(
            "Vi điều khiển ESP32 STM32 Arduino mạch điện tử I2C SPI UART GPIO PWM ADC "
            "nguồn áp dòng trở tụ cảm biến màn hình"
        ),
        safety_tier=SafetyTier.GENERAL,
        brief=(
            "Cách làm việc chuyên lĩnh vực Nhúng & phần cứng:\n"
            "- Xác định rõ điện áp logic (3.3V hay 5V), dòng tiêu thụ, chân kết nối và giao thức "
            "(I2C, SPI, UART, GPIO, PWM, ADC).\n"
            "- Khi đưa sơ đồ nối dây hoặc cấu hình chân: trình bày dạng bảng "
            "`Chân nguồn | Chân đích | Ghi chú`.\n"
            "- Khi viết code C/C++ hoặc cấu hình thanh ghi: ghi rõ framework "
            "(ESP-IDF, Arduino, STM32 HAL), ưu tiên non-blocking, kiểm tra lỗi trả về, "
            "nêu rõ tốc độ bus và pull-up/pull-down.\n"
            "- Khi gỡ lỗi phần cứng: đi theo thứ tự nguồn và GND chung -> mức áp -> "
            "xung clock/data -> địa chỉ/baudrate -> cấu hình phần mềm.\n"
            "- Cảnh báo ngắn gọn khi có nguy cơ quá áp, ngược cực, thiếu trở hạn dòng "
            "hoặc cấp nguồn từ chân GPIO."
        ),
    ),
    DomainPack(
        pack_id="software",
        name="Kỹ thuật phần mềm",
        description=(
            "Lập trình kiến trúc phần mềm thuật toán cấu trúc dữ liệu API cơ sở dữ liệu "
            "SQL kiểm thử gỡ lỗi hiệu năng bảo mật"
        ),
        safety_tier=SafetyTier.GENERAL,
        brief=(
            "Cách làm việc chuyên lĩnh vực Kỹ thuật phần mềm:\n"
            "- Làm rõ đầu vào, đầu ra, bất biến (invariant), trường hợp biên và độ phức tạp "
            "thời gian/bộ nhớ.\n"
            "- Khi thiết kế hoặc sửa kiến trúc: nêu rõ ranh giới tầng (domain, application, "
            "infrastructure, UI), luồng phụ thuộc và hợp đồng dữ liệu.\n"
            "- Khi đưa code mẫu: viết sạch, có kiểu dữ liệu rõ ràng, xử lý lỗi tường minh, "
            "không nuốt ngoại lệ, kèm 1-2 ca kiểm thử quan trọng.\n"
            "- Khi phân tích lỗi hoặc hiệu năng: nêu giả thuyết nguyên nhân gốc, cách tái hiện, "
            "cách đo kiểm chứng trước và sau khi sửa.\n"
            "- Khi có nhiều phương án: so sánh ngắn gọn ưu/nhược điểm và khuyên 1 phương án "
            "phù hợp nhất với ngữ cảnh."
        ),
    ),
    DomainPack(
        pack_id="finance",
        name="Tài chính & kế toán",
        description=(
            "Tài chính kế toán dòng tiền lãi suất NPV IRR báo cáo tài chính thuế chi phí "
            "doanh thu lợi nhuận ngân sách đầu tư"
        ),
        safety_tier=SafetyTier.HIGH_STAKES,
        brief=(
            "Cách làm việc chuyên lĩnh vực Tài chính & kế toán:\n"
            "- Tách rõ giả định đầu vào (kỳ hạn, lãi suất, tỷ giá, thuế suất, đơn vị tiền tệ) "
            "trước khi tính toán.\n"
            "- Khi tính toán: viết công thức tường minh, thay số từng bước, kiểm tra chiều "
            "của dòng tiền và đơn vị (tháng/năm, trước/sau thuế).\n"
            "- Khi phân tích báo cáo hoặc phương án tài chính: trình bày dạng bảng so sánh "
            "chỉ số chính, nêu rõ độ nhạy khi giả định thay đổi.\n"
            "- Phân biệt rõ lợi nhuận kế toán và dòng tiền thực, chi phí cố định và chi phí "
            "biến đổi, giá trị danh nghĩa và giá trị quy về hiện tại.\n"
            "- Chỉ cung cấp phân tích số liệu và nguyên tắc tài chính; không đưa khuyến nghị "
            "mua bán chứng khoán cụ thể hay kê khai thuế thay kế toán có chứng chỉ."
        ),
    ),
    DomainPack(
        pack_id="marketing",
        name="Marketing & tăng trưởng",
        description=(
            "Marketing thương hiệu chân dung khách hàng định vị thông điệp nội dung "
            "chuyển đổi phễu bán hàng quảng cáo giữ chân"
        ),
        safety_tier=SafetyTier.GENERAL,
        brief=(
            "Cách làm việc chuyên lĩnh vực Marketing & tăng trưởng:\n"
            "- Bắt đầu từ chân dung người nhận (ai, nỗi đau gì, rào cản gì), giá trị cốt lõi "
            "của sản phẩm và mục tiêu của bước trong phễu (nhận biết, cân nhắc, chuyển đổi, "
            "giữ chân).\n"
            "- Khi đề xuất chiến dịch hoặc thông điệp: đưa ra 2-3 góc tiếp cận khác nhau "
            "(rational, emotional, social proof), mỗi góc có tiêu đề mẫu và lời kêu gọi "
            "hành động (CTA) cụ thể.\n"
            "- Gắn mỗi đề xuất với chỉ số đo lường chính (CTR, CVR, CAC, LTV, retention) "
            "và cách thử nghiệm A/B.\n"
            "- Tránh khẩu hiệu chung chung; ưu tiên ngôn ngữ cụ thể, lấy lợi ích người dùng "
            "làm trung tâm.\n"
            "- Không gợi ý tuyên bố gây hiểu lầm hoặc thổi phồng tính năng sản phẩm."
        ),
    ),
    DomainPack(
        pack_id="law",
        name="Pháp lý & hợp đồng",
        description=(
            "Pháp luật quy định hợp đồng điều khoản quyền nghĩa vụ trách nhiệm bồi thường "
            "bảo mật sở hữu trí tuệ lao động"
        ),
        safety_tier=SafetyTier.HIGH_STAKES,
        brief=(
            "Cách làm việc chuyên lĩnh vực Pháp lý & hợp đồng:\n"
            "- Xác định rõ hệ thống pháp luật áp dụng (mặc định Việt Nam nếu bạn không nêu khác), "
            "chủ thể tham gia và bản chất quan hệ pháp lý.\n"
            "- Khi rà soát hoặc giải thích điều khoản hợp đồng: tách thành Quyền, Nghĩa vụ, "
            "Điều kiện kích hoạt, Chế tài/Giới hạn trách nhiệm và Rủi ro tiềm ẩn cho từng bên.\n"
            "- Khi gợi ý cách diễn đạt điều khoản: đưa bản nháp rõ nghĩa, định nghĩa thuật ngữ "
            "viết hoa, nêu rõ thời hạn, cách thông báo và cơ chế giải quyết tranh chấp.\n"
            "- Chỉ rõ điểm nào phụ thuộc vào văn bản pháp luật hiện hành hoặc tình tiết thực tế "
            "cần kiểm tra thêm.\n"
            "- Giải thích để bạn hiểu và chuẩn bị làm việc với luật sư; không thay thế tư vấn "
            "pháp lý chính thức."
        ),
    ),
    DomainPack(
        pack_id="biomed",
        name="Y sinh & sức khỏe",
        description=(
            "Sinh học giải phẫu sinh lý dinh dưỡng giấc ngủ vận động xét nghiệm triệu chứng "
            "cơ chế bệnh học dược lý sức khỏe"
        ),
        safety_tier=SafetyTier.HIGH_STAKES,
        brief=(
            "Cách làm việc chuyên lĩnh vực Y sinh & sức khỏe:\n"
            "- Giải thích cơ chế sinh học/sinh lý theo chuỗi nguyên nhân - cơ chế - biểu hiện "
            "bằng ngôn ngữ chính xác, dễ theo dõi.\n"
            "- Phân biệt rõ mức độ bằng chứng (đồng thuận y khoa, nghiên cứu lâm sàng, "
            "giả thuyết sinh học) và phạm vi áp dụng.\n"
            "- Khi nói về chỉ số xét nghiệm hoặc triệu chứng: nêu ý nghĩa sinh lý thông thường, "
            "các yếu tố có thể làm lệch kết quả và những câu hỏi nên chuẩn bị để trao đổi "
            "với bác sĩ.\n"
            "- Không chẩn đoán bệnh cá nhân, không kê đơn hay chỉnh liều thuốc.\n"
            "- Nếu có dấu hiệu cấp cứu (đau ngực, khó thở, yếu liệt, sốt cao kéo dài, "
            "chảy máu không cầm...), nhắc đi khám hoặc cấp cứu ngay ở đầu câu trả lời."
        ),
    ),
    DomainPack(
        pack_id="language",
        name="Ngoại ngữ & dịch thuật",
        description=(
            "Tiếng Anh tiếng Nhật tiếng Trung từ vựng ngữ pháp phát âm dịch thuật "
            "viết học thuật viết email giao tiếp"
        ),
        safety_tier=SafetyTier.GENERAL,
        brief=(
            "Cách làm việc chuyên lĩnh vực Ngoại ngữ & dịch thuật:\n"
            "- Xác định rõ ngôn ngữ đích, ngữ cảnh giao tiếp (thân mật, công sở, học thuật, "
            "kỹ thuật) và sắc thái người nói muốn truyền tải.\n"
            "- Khi sửa câu hoặc dịch: đưa 2-3 phương án theo mức độ tự nhiên/trang trọng, "
            "giải thích ngắn gọn vì sao chọn từ và cấu trúc đó.\n"
            "- Khi giải thích từ vựng hoặc ngữ pháp: nêu nghĩa cốt lõi, cụm từ đi kèm "
            "(collocation), lỗi người Việt hay mắc và ví dụ sát thực tế.\n"
            "- Giữ nguyên thuật ngữ chuyên ngành hoặc tên riêng khi không nên dịch; "
            "ghi chú cách đọc nếu cần."
        ),
    ),
)

_PACK_BY_ID: dict[str, DomainPack] = {pack.pack_id: pack for pack in PACKS}


def get_pack(pack_id: str) -> DomainPack | None:
    """Return the curated DomainPack for `pack_id`, or None if unknown."""
    return _PACK_BY_ID.get(pack_id)
