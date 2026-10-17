"""Greetings must be answered directly, never routed into retrieval."""

from __future__ import annotations

import pytest

from app.pipeline import DIRECT, GREETING_FALLBACK
from app.prompts import NOT_FOUND
from app.smalltalk import is_smalltalk, normalize


# --------------------------------------------------------------- detection
@pytest.mark.parametrize("text", [
    "halo", "Halo!", "HALO", "hai", "hi", "hello", "oi",
    "selamat pagi", "Selamat Pagi!", "pagi", "malam",
    "assalamualaikum", "Assalamu'alaikum", "salam",
    "makasih", "terima kasih", "Terima kasih banyak!", "thanks", "thank you",
    "oke", "ok", "oke makasih", "baik terima kasih", "sip",
    "bye", "sampai jumpa", "dadah", "cukup",
    "test", "tes", "coba", "apa kabar", "how are you",
    "siapa kamu", "kamu siapa", "Kamu siapa sih?", "who are you",
    "kamu bisa apa", "kamu bisa apa?", "apa yang bisa kamu bantu",
    "bisa bantu apa", "what can you do", "bisa bantu?", "tolong",
    "halo bot", "hai, selamat pagi", "halo halo",
    "",           # empty input: nothing to retrieve
    "???",        # punctuation only
    "👋",          # emoji only
])
def test_conversational_input_is_detected(text):
    assert is_smalltalk(text) is True


@pytest.mark.parametrize("text", [
    # Real questions, including ones wrapped in a greeting.
    "halo, berapa total SKS semester 3?",
    "Hai, apa syarat mengambil skripsi?",
    "selamat pagi, bagaimana prosedur pengajuan KRS?",
    "makasih, tapi berapa biaya kuliahnya?",
    "berapa SKS",
    "apa itu MK pengayaan",
    "jelaskan kurikulum teknik informatika",
    "siapa yang menandatangani ijazah",
    "daftar mata kuliah semester 1",
    "kapan jadwal ujian",
    "what is the curriculum",
    "list all courses",
    # Long input is never a greeting, even without an obvious cue word.
    "tolong buatkan ringkasan lengkap mengenai seluruh isi dokumen panduan akademik ini",
])
def test_real_questions_are_not_treated_as_smalltalk(text):
    assert is_smalltalk(text) is False


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Halo, Dunia!! ") == "halo dunia"


# ------------------------------------------------------------ in pipeline
def test_greeting_skips_retrieval_and_answers_directly(service):
    result = service.ask("halo")

    assert result.route == DIRECT
    assert result.route_method == "smalltalk"
    assert result.documents == []          # nothing retrieved
    assert result.graph is None
    assert result.answer != NOT_FOUND      # the actual bug being fixed
    assert result.answered is False        # answered, but not from sources
    # Neither the router nor the retriever ran, so neither is timed.
    assert "router" not in result.timings
    assert "vector" not in result.timings


def test_greeting_does_not_call_the_router(service):
    """The whole point of the rule-based check: no extra LLM round-trip."""
    before = service.llm.calls
    service.ask("makasih")
    # Exactly one call: the smalltalk reply itself.
    assert service.llm.calls == before + 1


def test_real_question_still_retrieves(service):
    result = service.ask("Bagaimana prosedur pengajuan cuti?")

    assert result.route != DIRECT
    assert result.documents
    assert result.answered is True


def test_forced_route_overrides_smalltalk_detection(service):
    """An explicit route means the caller wants retrieval; honour it."""
    result = service.ask("halo", route_override="VECTOR")

    assert result.route == "VECTOR"
    assert result.route_method == "manual"


def test_greeting_streams_tokens_and_reports_direct_route(service):
    events = list(service.stream("hai"))
    kinds = [k for k, _ in events]

    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert "token" in kinds
    done = events[-1][1]
    assert done.route == DIRECT
    assert done.answered is False
    assert done.answer and done.answer != NOT_FOUND


def test_empty_model_reply_to_greeting_falls_back(tmp_path):
    """A local model sometimes returns whitespace; an empty bubble looks broken."""
    from app.service import RAGService
    from tests.conftest import make_settings
    from tests.fakes import FakeChatModel, HashingEmbeddings

    llm = FakeChatModel(responder=lambda messages: "   ")
    svc = RAGService(make_settings(tmp_path), llm=llm, embeddings=HashingEmbeddings())
    try:
        svc.ingest(recreate=True)
        assert svc.ask("halo").answer == GREETING_FALLBACK
        streamed = "".join(p for k, p in svc.stream("halo") if k == "token")
        assert streamed == GREETING_FALLBACK
    finally:
        svc.close()


def test_not_found_answer_is_marked_unanswered(service):
    """`answered` must distinguish 'no context' from 'answered from context'."""
    result = service.ask("Berapa harga tiket pesawat ke Tokyo?", route_override="VECTOR")

    # The fixture corpus has nothing about this; whatever is retrieved, the
    # flag has to agree with whether context was actually found.
    assert result.answered == bool(result.documents)
