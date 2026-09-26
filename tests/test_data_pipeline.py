from data.dedup import Deduplicator
from data.langid import LanguageIdentifier
from data.normalize import TextNormalizer, dedup_key
from data.quality import LineFilter, QualityFilter
from data.tanglish import Transliterator, colloquialize


def test_nfc_composes_split_vowel_signs():
    n = TextNormalizer()
    assert n("நொடி") == "நொடி"          # ெ + ா → ொ
    assert n("பௌர்ணமி") == "பௌர்ணமி"      # ெ + ௗ → ௌ


def test_invisible_and_tamil_sign_repairs():
    n = TextNormalizer()
    assert n("தமி‍ழ்") == "தமிழ்"
    assert n("காா") == "கா"
    assert n("க்ா") == "கா"
    assert n("ஶ்ரீ") == "ஸ்ரீ"
    assert n("a  b\r\n\n\n\nc") == "a b\n\nc"


def test_pii_masking_and_urls():
    n = TextNormalizer({"urls": "remove", "emails": "mask", "phones": "mask"})
    out = n("mail me at a.b@example.com or 9876543210, see https://x.org/y")
    assert "<email>" in out and "<phone>" in out and "https" not in out


def test_dedup_key_keeps_tamil_marks():
    assert dedup_key("கல்") != dedup_key("கல")


def test_langid_labels():
    lid = LanguageIdentifier()
    assert lid("தமிழ்நாட்டின் தலைநகரம் சென்னை. இது ஒரு பெரிய நகரம்.").label == "ta"
    assert lid("Naan inniki office ku late ah varen, traffic romba jaasthi ah iruku da.").label == "tanglish"
    assert lid("The quick brown fox jumps over the lazy dog and it was a sunny day.").label == "en"
    assert lid("இந்த படம் பார்த்தேன், the climax was really amazing and the music too").label == "code_mixed"
    assert lid("Der schnelle braune Fuchs springt über den faulen Hund im Park.").label == "other"


def test_transliteration_contextual_rules():
    t = Transliterator()
    assert [t.word(w) for w in "தமிழ் வணக்கம் நன்றி வெற்றி பச்சை மஞ்சள் வந்து எங்க".split()] == [
        "thamizh", "vanakkam", "nandri", "vetri", "pachai", "manjal", "vandhu", "enga"]


def test_colloquialize():
    assert colloquialize("நான் போகிறேன்") == "நான் போறேன்"
    assert colloquialize("அவர்கள் படிக்கிறார்கள்") == "அவங்க படிக்குறாங்க"
    assert colloquialize("எனக்கு தெரியவில்லை") == "எனக்கு தெரியல"


def test_quality_rejects_repetition_and_short():
    q = QualityFilter({"default": {"min_chars": 50, "min_words": 5}})
    ok, reason, _ = q.check("சிறிய", "ta")
    assert not ok and reason == "too_short_chars"
    ok, reason, _ = q.check(" ".join(["ஒன்று இரண்டு மூன்று"] * 60), "ta")
    assert not ok
    ok, _, _ = q.check("தமிழ் ஒரு பழமையான மொழி. இதற்கு நீண்ட இலக்கிய வரலாறு உண்டு. "
                       "சங்க இலக்கியம் இரண்டாயிரம் ஆண்டுகளுக்கு மேல் பழமையானது.", "ta")
    assert ok


def test_line_filter_drops_boilerplate_and_repeats():
    lf = LineFilter()
    out = lf("முதல் வரி\nAll rights reserved 2024\nமுதல் வரி\nஇரண்டாம் வரி")
    assert out == "முதல் வரி\nஇரண்டாம் வரி"


def test_dedup_exact_and_near():
    d = Deduplicator({"bands": 10, "rows": 12, "verify_threshold": 0.7})
    base = " ".join(f"சொல்{i}" for i in range(200))
    assert d.is_duplicate(base) is None
    assert d.is_duplicate(base + " ") == "exact"
    near = base.replace("சொல்5 ", "மாற்றம் ")
    assert d.is_duplicate(near) == "near"
    assert d.is_duplicate(" ".join(f"வேறு{i}" for i in range(200))) is None


def test_strip_prefixes():
    from data.sources import strip_prefixes

    assert strip_prefixes("சூழல்: ஆரம்பத்தில்", ["சூழல்:"]) == "ஆரம்பத்தில்"
    assert strip_prefixes("ஆரம்பத்தில்", ["சூழல்:"]) == "ஆரம்பத்தில்"


def test_qa_format_and_unanswerable():
    from training.chat_data import row_to_messages

    fields = {"context": "Context", "question": "Question", "answer": "Answer"}
    row = {"Context": "சூழல்: சென்னை தமிழ்நாட்டின் தலைநகரம்.", "Question": "தலைநகரம் எது?", "Answer": "சென்னை"}
    msgs = row_to_messages(row, "qa", fields, {"strip_prefixes": ["சூழல்:"]})
    assert "பத்தி: சென்னை" in msgs[0]["content"] and "சூழல்" not in msgs[0]["content"]
    assert msgs[1]["content"] == "சென்னை"
    unanswerable = {**row, "Answer": None}
    assert row_to_messages(unanswerable, "qa", fields, {"unanswerable_response": None}) is None
    assert row_to_messages(unanswerable, "qa", fields, {"unanswerable_response": "இல்லை"})[1]["content"] == "இல்லை"
    assert row_to_messages({**row, "Context": None}, "qa", fields, {}) is None
