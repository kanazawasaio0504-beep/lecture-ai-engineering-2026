"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_save_learning_log_uses_a_timestamp_filename_and_markdown(monkeypatch, tmp_path):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)

    message = tools.save_learning_log(
        topic="../LoRA と QLoRA",
        summary="- LoRA: 低ランク行列で追加学習する。\n- QLoRA: 量子化したモデルを使う。",
        lecture=3,
    )

    files = list(log_dir.glob("*.md"))
    assert len(files) == 1
    assert files[0].name != "../LoRA と QLoRA.md"
    assert files[0].stem[:8].isdigit()
    assert message == f"学習の記録を保存しました: learning_log/{files[0].name}"
    content = files[0].read_text(encoding="utf-8")
    assert "# ../LoRA と QLoRA" in content
    assert "- 関係する回: 第3回" in content
    assert "LoRA:" in content and "QLoRA:" in content


def test_make_quiz_selects_a_term_from_the_requested_lecture(monkeypatch):
    terms = [entry for entry in tools.load_glossary() if entry["lecture"] == 2]
    monkeypatch.setattr(tools.random, "choice", lambda choices: choices[0])

    quiz = json.loads(tools.make_quiz(2))

    assert quiz == {
        "question": f"次の説明に当てはまる用語は何ですか？ {terms[0]['definition']}",
        "answer": terms[0]["term"],
    }


def test_make_quiz_reports_when_lecture_has_no_terms():
    assert tools.make_quiz(99) == "第99回の用語が見つかりませんでした。"


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"
