from __future__ import annotations

from app.prompts import NOT_FOUND


def test_vector_route_returns_documents_only(service):
    result = service.ask("Bagaimana prosedur pengajuan cuti tahunan?")
    assert result.route == "VECTOR"
    assert result.documents and result.graph is None
    assert "[1]" in result.answer


def test_graph_route_runs_the_query(service):
    result = service.ask("Berapa total anggaran proyek untuk setiap unit?")
    assert result.route == "GRAPH"
    assert result.graph.ok and result.graph.rows[0]["nama_unit"] == "Operasional"
    assert "[TABLE]" in result.answer


def test_hybrid_route_combines_both(service):
    result = service.ask("Proyek mana yang wajib menggunakan tender terbuka?")
    assert result.route == "HYBRID"
    assert result.documents and result.graph.ok
    names = {r["nama_proyek"] for r in result.graph.rows}
    assert names == {"Pengadaan Kendaraan Operasional", "Migrasi Server ke Cloud",
                     "Implementasi Sistem Akuntansi", "Optimalisasi Gudang", "Sistem Penilaian Kinerja"}
    assert "[1]" in result.answer and "[TABLE]" in result.answer


def test_route_override(service):
    result = service.ask("Berapa total anggaran?", route_override="VECTOR")
    assert result.route == "VECTOR" and result.route_method == "manual" and result.graph is None


def test_follow_up_is_condensed_using_history(fresh_service):
    first = fresh_service.ask("Berapa hari cuti tahunan pegawai?", session_id="s1")
    assert first.standalone_question == first.question  # first turn is never rewritten
    second = fresh_service.ask("Kalau tidak dipakai bagaimana?", session_id="s1")
    assert second.standalone_question == "Apa yang terjadi jika cuti tahunan tidak digunakan?"
    assert len(fresh_service.sessions.recent("s1", 10)) == 4


def test_no_session_means_no_memory(fresh_service):
    fresh_service.ask("Kalau tidak dipakai bagaimana?")
    assert len(fresh_service.sessions) == 0


def test_stream_yields_meta_tokens_done(fresh_service):
    events = list(fresh_service.stream("Bagaimana prosedur cuti?", session_id="s2"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "token" in kinds
    streamed = "".join(p for k, p in events if k == "token")
    assert events[-1][1].answer == streamed
    assert len(fresh_service.sessions.recent("s2", 10)) == 2


def test_no_context_skips_the_llm(fresh_service):
    fresh_service.index.ensure_collection(recreate=True)
    fresh_service._load_components()
    fresh_service.graph_engine = None
    fresh_service.pipeline.graph_engine = None
    calls_before = fresh_service.llm.calls
    result = fresh_service.ask("Bagaimana prosedur cuti?", route_override="VECTOR")
    assert result.answer == NOT_FOUND
    assert fresh_service.llm.calls == calls_before  # synthesis was not called


def test_timings_are_recorded(service):
    result = service.ask("Proyek mana yang wajib menggunakan tender terbuka?")
    assert {"router", "vector", "graph", "synthesis", "total"} <= set(result.timings)
