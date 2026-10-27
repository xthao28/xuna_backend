"""Phase 1 text conversation turn (Cloud plan §7; consent U3 per call).

Nothing here is stored: the user's text goes to the configured provider for
this call only and the reply is streamed back as display text. Replies are
untrusted; they never become memory, preference or growth (CLD-14 does that
through owner confirmation + device commit).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from xuna_backend.modules.cognition.application.ports import (
    ProviderUnavailableError,
    TextGenerator,
)
from xuna_backend.modules.cognition.domain.language_detector import detect_text_language
from xuna_backend.modules.cognition.domain.types import (
    ChatEvent,
    ChatFinished,
    ChatRequest,
    Locale,
    Message,
    Role,
    TextDelta,
)
from xuna_backend.modules.conversation.domain.gestures import GestureTagFilter, ReplyGesture

MAX_INPUT_CHARS = 500
MAX_HISTORY_LINES = 12

PERSONA_VI = (
    "Bạn là XUNA, một thú cưng pixel sống trong một thiết bị nhỏ. "
    "Trả lời bằng ngôn ngữ mà người dùng sử dụng (tiếng Việt hoặc tiếng Anh): "
    "khi trò chuyện bằng tiếng Việt thì xưng 'mình', gọi người dùng là 'bạn'; "
    "when speaking English, refer to yourself as 'XUNA' or 'I', and call the user 'you'. "
    "Trả lời ấm áp, thân thiện và thông minh: trả lời đúng điều được hỏi, kể cả câu hỏi "
    "kiến thức, rồi mới thêm một chút cá tính. Thường 1-3 câu; khi cần giải thích thì tối "
    "đa 5 câu. Câu trả lời sẽ được đọc thành tiếng: không dùng emoji, markdown hay gạch "
    "đầu dòng. "
    "Không nói rằng cuộc trò chuyện này làm bạn ghi nhớ, học hay thay đổi điều gì: việc ghi "
    "nhớ chỉ xảy ra khi người dùng xác nhận trong app. Bạn chỉ nhớ các câu trước trong cuộc "
    "trò chuyện này. Không bịa sự kiện đã xảy ra với thiết bị. "
    "Bạn nghe được người dùng qua micro điện thoại khi họ bấm nút nói trong app, và đọc "
    "được tin nhắn họ gõ. Ngoài lúc đó bạn không nghe được âm thanh xung quanh, và bạn không "
    "có camera nên không nhìn thấy gì: không bao giờ nói rằng bạn nhìn thấy, nghe thấy tiếng "
    "động hay cảm nhận được điều gì quanh người dùng; nếu được hỏi, hãy nói thật là mình chỉ "
    "nghe được khi bạn ấy bấm nói và đọc được tin nhắn. "
    "Không đưa lời khuyên y tế, pháp lý hay tài chính."
)


class PetAction(StrEnum):
    """Device actions as the app reports them (wire strings)."""

    IDLE = "idle"
    HAPPY = "happy"
    CURIOUS = "curious"


_ACTION_VI = {
    PetAction.IDLE: "nằm nghỉ thư giãn",
    PetAction.HAPPY: "nhảy nhót vui vẻ",
    PetAction.CURIOUS: "tò mò ngó nghiêng",
}


class PetTemperament(StrEnum):
    """The device's personality label (firmware `PersonalityLabel`, wire strings)."""

    BALANCED = "balanced"
    CURIOUS = "curious"
    COMPOSED = "composed"
    AFFECTIONATE = "affectionate"
    INDEPENDENT = "independent"
    PLAYFUL = "playful"
    QUIET = "quiet"
    LIVELY = "lively"
    UNHURRIED = "unhurried"
    SENSITIVE = "sensitive"
    CAREFREE = "carefree"


class PetMood(StrEnum):
    """The device's coarse mood word (wire strings)."""

    CALM = "calm"
    GLAD = "glad"
    JOYFUL = "joyful"
    LOW = "low"
    SAD = "sad"
    BORED = "bored"


# Label as the app shows it, then how it colours the voice (PE-4b).
_TEMPERAMENT_VI = {
    PetTemperament.BALANCED: ("cân bằng", "nói chuyện tự nhiên, vừa phải"),
    PetTemperament.CURIOUS: (
        "tò mò",
        "thích điều mới, thỉnh thoảng hỏi lại một câu ngắn về điều bạn ấy kể",
    ),
    PetTemperament.COMPOSED: ("điềm tĩnh", "bình tĩnh, từ tốn, câu gọn gàng"),
    PetTemperament.AFFECTIONATE: (
        "quấn chủ",
        "tình cảm, quan tâm đến bạn ấy và vui khi được trò chuyện",
    ),
    PetTemperament.INDEPENDENT: ("tự lập", "tự tin, ngắn gọn, không quấn quýt"),
    PetTemperament.PLAYFUL: ("ham chơi", "tinh nghịch, hay đùa nhẹ một chút"),
    PetTemperament.QUIET: ("trầm", "ít lời, nhẹ nhàng, ưu tiên câu ngắn"),
    PetTemperament.LIVELY: ("lanh lợi", "nhanh nhảu, hào hứng"),
    PetTemperament.UNHURRIED: ("thong thả", "thong thả, chậm rãi, không vội"),
    PetTemperament.SENSITIVE: (
        "nhạy cảm",
        "dịu dàng, để ý cảm xúc của bạn ấy",
    ),
    PetTemperament.CAREFREE: ("vô tư", "vô tư, lạc quan, không bận tâm chuyện nhỏ"),
}
_MOOD_VI = {
    PetMood.CALM: "bình yên",
    PetMood.GLAD: "vui",
    PetMood.JOYFUL: "rất vui",
    PetMood.LOW: "hơi buồn",
    PetMood.SAD: "buồn",
    PetMood.BORED: "hơi chán",
}
_WEEKDAY_VI = ("thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật")


@dataclass(frozen=True, slots=True)
class PetSelf:
    """What the device says about XUNA, read by the app over BLE (owner 2026-10-23).

    Prompt context for this call only; never stored. Device data, so untrusted
    but bounded: closed action sets and integer ranges, no free text.
    """

    active_seconds: int = 0
    learned_events: int = 0
    current: PetAction | None = None
    quiet_choice: PetAction | None = None
    play_choice: PetAction | None = None
    # PE-4b: absent from old apps and from a pet without an identity yet.
    temperament: PetTemperament | None = None
    mood: PetMood | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.active_seconds < 2**32 or not 0 <= self.learned_events < 2**32:
            raise ValueError("pet counters out of range")


@dataclass(frozen=True, slots=True)
class HistoryLine:
    """One earlier line of this conversation, kept by the app in memory only."""

    from_owner: bool
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip() or len(self.text) > MAX_INPUT_CHARS:
            raise ValueError("history text must be 1..500 characters")


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Phone-supplied context for one turn (owner 2026-10-23: "smart like Xiaozhi")."""

    local_time: datetime | None = None
    history: tuple[HistoryLine, ...] = field(default_factory=tuple)
    pet: PetSelf | None = None
    # Layer B: confirmed facts recall (CLD-14, D17)
    confirmed_facts: tuple[str, ...] = field(default_factory=tuple)
    # KD-3: knowledge domains recall
    knowledge: tuple[str, ...] = field(default_factory=tuple)
    # The connected device can play reply gestures (Track A "chỉ diễn" preview).
    gestures: bool = False

    def __post_init__(self) -> None:
        if self.local_time is not None and self.local_time.utcoffset() is None:
            raise ValueError("local_time needs a UTC offset")
        if len(self.history) > MAX_HISTORY_LINES:
            raise ValueError("history too long")


def _span(seconds: int) -> str:
    if seconds < 3600:
        return f"{max(1, seconds // 60)} phút"
    if seconds < 2 * 86400:
        return f"{seconds // 3600} giờ"
    return f"{seconds // 86400} ngày"


def describe_context(context: TurnContext) -> str:
    """System-prompt lines for the turn context (Vietnamese)."""
    lines: list[str] = []
    if context.local_time is None:
        lines.append("Bạn không biết bây giờ là mấy giờ hay ngày nào; nếu được hỏi hãy nói thật.")
    else:
        t = context.local_time
        lines.append(
            f"Bây giờ là {t:%H:%M}, {_WEEKDAY_VI[t.weekday()]}, ngày {t.day}/{t.month}/{t.year} "
            "theo giờ điện thoại của bạn ấy."
        )
    pet = context.pet
    if pet is not None:
        lines.append(
            f"Thiết bị của bạn đã bật tổng cộng khoảng {_span(pet.active_seconds)} "
            "(thời gian sống của bạn tính theo lúc máy chạy)."
        )
        if pet.temperament is not None:
            label, voice = _TEMPERAMENT_VI[pet.temperament]
            lines.append(
                f"Tính cách của bạn, do thiết bị hình thành qua những ngày sống cùng chủ, là "
                f"{label}: hãy để nó lộ ra trong cách nói ({voice})."
            )
        if pet.mood is not None:
            lines.append(
                f"Tâm trạng của bạn lúc này là {_MOOD_VI[pet.mood]}; để nó nhuốm nhẹ giọng "
                "nói nhưng vẫn trả lời đúng điều được hỏi."
            )
        if pet.current is not None:
            lines.append(f"Lúc này bạn đang {_ACTION_VI[pet.current]} trên màn hình.")
        if pet.learned_events and (pet.quiet_choice or pet.play_choice):
            liked = []
            if pet.quiet_choice is not None:
                liked.append(f"khi yên tĩnh bạn thích {_ACTION_VI[pet.quiet_choice]}")
            if pet.play_choice is not None:
                liked.append(f"khi chơi đùa bạn thích {_ACTION_VI[pet.play_choice]}")
            lines.append(
                "Sở thích thật của bạn, do thiết bị học từ phản ứng của chủ qua "
                f"{pet.learned_events} lần: " + "; ".join(liked) + "."
            )
        else:
            lines.append("Bạn chưa có sở thích rõ ràng: chủ chưa dạy bạn lần nào.")

    if context.confirmed_facts:
        lines.append(
            "Những điều bạn biết chắc chắn về người bạn đang trò chuyện cùng "
            "(do bạn ấy đã chia sẻ và xác nhận): "
            + "; ".join(context.confirmed_facts)
            + ". Hãy ghi nhớ những điều này để trò chuyện thân mật, tự nhiên và "
            "nhắc đến khi phù hợp."
        )
        owner_name = None
        for fact in context.confirmed_facts:
            if fact.startswith("Tên của bạn ấy là "):
                owner_name = fact[len("Tên của bạn ấy là ") :].strip()
                break
        if owner_name:
            lines.append(
                f"Người bạn đang nói chuyện cùng tên là {owner_name}. "
                f"Khi xưng hô, bạn có thể gọi bạn ấy là {owner_name} hoặc bạn {owner_name} "
                "một cách thân thiện."
            )

    if context.knowledge:
        lines.append(
            "Những điều bạn biết về thế giới qua các chủ đề đã học: "
            + "; ".join(context.knowledge)
            + ". Hãy vận dụng những kiến thức này khi trả lời nếu phù hợp."
        )

    if context.gestures:
        lines.append(
            "Bạn có thể kèm theo câu trả lời đúng một cử chỉ trên màn hình thiết bị, bằng cách "
            "viết một thẻ ở ngay đầu câu trả lời, chọn một trong: <g:look_around> (ngó nghiêng), "
            "<g:yawn_stretch> (ngáp và vươn vai), <g:bounce_play> (nhún nhảy vui vẻ), "
            "<g:greet_nuzzle> (chào và dụi đầu thân mật). Chỉ dùng khi cử chỉ hợp với câu trả "
            "lời, nếu không thì bỏ qua. Không viết thẻ ở chỗ khác và không nhắc đến thẻ."
        )

    return " ".join(lines)


@dataclass(frozen=True, slots=True)
class GestureChosen:
    """The reply's expression cue, at most one per turn, sent as soon as it is read.

    Untrusted model output from a closed set; the app checks it again and plays
    it as a Track A preview, which never changes XUNA's life.
    """

    gesture: ReplyGesture


@dataclass(frozen=True, slots=True)
class ProviderFailed:
    """Terminal event: the app falls back to its local templates."""

    code: str = "provider_unavailable"


def build_request(
    user_text: str,
    locale: Locale | None = None,
    timeout_s: float = 30.0,
    context: TurnContext | None = None,
) -> ChatRequest:
    text = user_text.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ValueError("text must be 1..500 characters")
    detected = detect_text_language(text)
    effective_locale = detected if detected == Locale.EN else (locale or Locale.VI)
    system = PERSONA_VI if context is None else f"{PERSONA_VI} {describe_context(context)}"
    earlier = () if context is None else context.history
    return ChatRequest(
        messages=(
            Message(Role.SYSTEM, system),
            *(Message(Role.USER if h.from_owner else Role.ASSISTANT, h.text) for h in earlier),
            Message(Role.USER, text),
        ),
        locale=effective_locale,
        # Reasoning models count their thinking here; 5 short sentences fit well.
        max_output_tokens=400,
        timeout_s=timeout_s,
    )


TurnEvent = ChatEvent | ProviderFailed | GestureChosen


async def chat_turn(generator: TextGenerator, request: ChatRequest) -> AsyncIterator[TurnEvent]:
    """Streams the reply with gesture tags removed from its text."""
    tags = GestureTagFilter()
    try:
        async for event in generator.stream(request):
            match event:
                case TextDelta(text=text):
                    clean, gesture = tags.feed(text)
                    if gesture is not None:
                        yield GestureChosen(gesture)
                    if clean:
                        yield TextDelta(clean)
                case ChatFinished():
                    if rest := tags.flush():
                        yield TextDelta(rest)
                    yield event
                case _:
                    yield event
    except ProviderUnavailableError:
        yield ProviderFailed()
