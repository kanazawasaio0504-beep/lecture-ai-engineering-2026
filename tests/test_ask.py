from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_counts_and_returns_response_text(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    result = ask_module.ask("質問")

    assert result == "回答"
    assert capsys.readouterr().out.splitlines() == [
        "[入力トークン数] 12",
        "[出力トークン数] 34",
    ]