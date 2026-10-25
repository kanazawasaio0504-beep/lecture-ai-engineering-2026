from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_counts_and_returns_response_text(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
            thoughts_token_count=5,
        ),
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
        "[思考トークン数] 5",
    ]


def test_ask_shows_unknown_when_thought_token_count_is_unavailable(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
            thoughts_token_count=None,
        ),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)

    ask_module.ask("質問")

    assert capsys.readouterr().out.splitlines() == [
        "[入力トークン数] 12",
        "[出力トークン数] 34",
        "[思考トークン数] 不明",
    ]


def test_ask_sends_previous_turns_with_follow_up_question(monkeypatch):
    responses = iter([
        SimpleNamespace(
            usage_metadata=None,
            text="最初の回答",
            candidates=[SimpleNamespace(content={
                "role": "model",
                "parts": [{"text": "最初の回答", "thought_signature": "signature"}],
            })],
        ),
        SimpleNamespace(
            usage_metadata=None,
            text="続きの回答",
            candidates=[SimpleNamespace(content={
                "role": "model", "parts": [{"text": "続きの回答"}]
            })],
        ),
    ])
    requests = []

    def generate_content(**kwargs):
        requests.append([dict(message) for message in kwargs["contents"]])
        return next(responses)

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    history = []

    ask_module.ask("最初の質問", history)
    ask_module.ask("続きの質問", history)

    assert requests == [
        [{"role": "user", "parts": [{"text": "最初の質問"}]}],
        [
            {"role": "user", "parts": [{"text": "最初の質問"}]},
            {
                "role": "model",
                "parts": [{"text": "最初の回答", "thought_signature": "signature"}],
            },
            {"role": "user", "parts": [{"text": "続きの質問"}]},
        ],
    ]


def test_main_accepts_follow_up_questions_until_blank(monkeypatch, capsys):
    calls = []

    def fake_ask(question, history):
        calls.append((question, history))
        return f"{question}への回答"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    answers = iter(["続きの質問", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    ask_module.main()

    assert [question for question, _ in calls] == ["最初の質問", "続きの質問"]
    assert calls[0][1] is calls[1][1]
    assert "続きの質問への回答" in capsys.readouterr().out