"""Detect conversational turns that must not be sent into retrieval.

Why this exists: the pipeline's job is to answer from documents, so every input
used to be condensed, routed, and searched. For "halo" that produces a search
with no relevant hits, and the honest-but-useless reply "Informasi tersebut
tidak ditemukan dalam basis pengetahuan." A user reading that concludes the bot
is broken, when in fact it was never asked a question.

The check is deliberately a rule, not an LLM call:
  * an extra LLM round-trip on every message is the slowest stage in the
    pipeline (see the latency figure in the evaluation), and
  * a local 4B model classifying "is this smalltalk" is less reliable than a
    pattern, while being impossible to reproduce in the thesis.

Conservative by design: the cost of missing a greeting is one unnecessary
search, but the cost of a false positive is refusing to answer a real question.
So anything carrying question content falls through to retrieval.
"""

from __future__ import annotations

import re

# Phrases that are *only* ever conversational. Matched against the whole input
# (after stripping punctuation), never as substrings of a longer question.
_EXACT = {
    # Indonesian greetings
    "halo", "hai", "hello", "helo", "hey", "hi", "oi", "woi",
    "pagi", "selamat pagi", "siang", "selamat siang", "sore", "selamat sore",
    "malam", "selamat malam", "selamat datang",
    "assalamualaikum", "assalamu alaikum", "assalammualaikum",
    "asalamualaikum", "salam", "waalaikumsalam", "walaikumsalam",
    # thanks / acknowledgement
    "makasih", "makasi", "terima kasih", "terimakasih", "thanks", "thank you",
    "thx", "tengkyu", "oke", "ok", "okay", "oke deh", "sip", "baik", "noted",
    "mantap", "keren", "bagus", "wah", "oh", "ohh", "i see", "got it",
    # goodbyes
    "bye", "byee", "sampai jumpa", "sampai nanti", "dadah", "selesai", "udahan",
    "cukup", "that is all", "thats all", "no", "tidak", "gak", "engga", "nggak",
    # tests
    "test", "tes", "testing", "coba", "ping", "apa kabar", "how are you",
    "kabar baik", "baik baik",
}

# Patterns for conversational turns that vary too much to enumerate.
# Anchored to the full input so a real question is never swallowed.
_PATTERNS = (
    # "siapa kamu", "kamu siapa", "kamu siapa sih"
    re.compile(r"^(siapa|sapa)\s+(kamu|kau|anda|lo|lu|km)\b.*$"),
    re.compile(r"^(kamu|kau|anda|lo|lu|km)\s+(siapa|sapa)\b.*$"),
    re.compile(r"^(who|what)\s+are\s+you\b.*$"),
    # "kamu bisa apa", "apa yang bisa kamu bantu", "bisa bantu apa"
    re.compile(r"^(kamu|kau|anda|lo|lu|km)\s+bisa\s+(apa|ngapain)\b.*$"),
    re.compile(r"^apa\s+(yang\s+)?(bisa|boleh)\s+(kamu|kau|anda|saya|aku)\b.*$"),
    re.compile(r"^bisa\s+(bantu\s+)?apa\b.*$"),
    re.compile(r"^(what|how)\s+can\s+you\s+(do|help)\b.*$"),
    # "kamu bisa bantu saya?", "bisa bantu?", "tolong bantu"
    re.compile(r"^((kamu|kau|anda)\s+)?(bisa|boleh)\s+(bantu|tolong)(\s+(saya|aku|gue|dong|ga|gak|nggak))*$"),
    re.compile(r"^(tolong|help)(\s+(saya|aku|me|dong))?$"),
    # greeting + filler only: "halo bot", "hai, selamat pagi"
    re.compile(r"^(halo|hai|hi|hello|helo|hey)\b[\s,]*(bot|pak|bu|kak|bang|min|admin|semua|guys)?$"),
    # "makasih banyak", "terima kasih ya", "makasih infonya"
    re.compile(r"^(makasih|makasi|terima\s*kasih|terimakasih|thanks|thank\s*you)\b"
               r"[\s,]*(banyak|ya|yah|kak|bang|bu|pak|infonya|info|bro|min)*$"),
    # pure acknowledgement: "oke makasih", "ok sip", "baik terima kasih"
    re.compile(r"^(oke?|ok|okay|baik|sip|siap)\b[\s,]*"
               r"(makasih|makasi|terima\s*kasih|terimakasih|thanks|deh|ya|bang|kak)*$"),
)

# If any of these appear, it is a real question regardless of the greeting
# wrapped around it: "halo, berapa total SKS semester 3?" must be answered.
_QUESTION_CUES = (
    "berapa", "bagaimana", "gimana", "apakah", "kapan", "dimana", "di mana",
    "siapa yang", "mengapa", "kenapa", "jelaskan", "sebutkan", "jumlah",
    "total", "daftar", "syarat", "prosedur", "cara", "aturan", "kebijakan",
    "biaya", "sks", "mata kuliah", "matakuliah", "semester", "kurikulum",
    "dosen", "prodi", "skripsi", "kuliah", "nilai", "ipk", "krs", "wajib",
    # English cues are kept narrow on purpose: a bare "how" would also match
    # "how are you", and a bare "what" would match "what are you".
    "how many", "how much", "how do", "how to", "how long",
    "what is", "what are the", "when", "where", "why", "explain", "list",
    "requirement",
)

# Greeting words that combine freely: "selamat pagi", "hai selamat pagi".
# "selamat" alone is not a greeting ("selamat" + what?), so it only counts
# inside such a sequence.
_GREETING_WORDS = {
    "halo", "hai", "hi", "hello", "helo", "hey", "oi", "woi", "salam",
    "selamat", "pagi", "siang", "sore", "malam", "datang",
    "assalamualaikum", "assalamu", "alaikum", "waalaikumsalam",
    "bot", "pak", "bu", "kak", "bang", "min", "admin", "semua",
}

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")

# Longer than this and it is not a greeting, whatever it looks like.
MAX_SMALLTALK_CHARS = 60


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and emoji, collapse whitespace."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def is_smalltalk(question: str) -> bool:
    """True when the input is conversational and carries no document question.

    >>> is_smalltalk("halo")
    True
    >>> is_smalltalk("halo, berapa total SKS semester 3?")
    False
    """
    text = normalize(question)
    if not text:
        return True  # empty/punctuation-only: nothing to retrieve
    if len(text) > MAX_SMALLTALK_CHARS:
        return False
    # A real question anywhere in the input wins over any greeting around it.
    if any(cue in text for cue in _QUESTION_CUES):
        return False
    if text in _EXACT:
        return True
    if any(pattern.match(text) for pattern in _PATTERNS):
        return True
    # Every word is a greeting token: "halo halo", "hai selamat pagi".
    words = text.split()
    if 1 <= len(words) <= 5 and all(w in _GREETING_WORDS or w in _EXACT for w in words):
        return True
    return False
