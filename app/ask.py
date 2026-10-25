"""第1回: 素のLLMと会話するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    回答後に続けて質問でき、空行で終了する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[dict] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    if history is None:
        history = []
    history.append({"role": "user", "parts": [{"text": question}]})
    response = client.models.generate_content(model=MODEL, contents=history)
    candidates = getattr(response, "candidates", None)
    model_content = candidates[0].content if candidates else None
    history.append(
        model_content
        or {"role": "model", "parts": [{"text": response.text or ""}]}
    )

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else "不明"
    output_tokens = usage.candidates_token_count if usage is not None else "不明"
    thought_tokens = usage.thoughts_token_count if usage is not None else None
    if thought_tokens is None:
        thought_tokens = "不明"
    print(f"[入力トークン数] {input_tokens}")
    print(f"[出力トークン数] {output_tokens}")
    print(f"[思考トークン数] {thought_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    history = []
    question = args.question
    try:
        while question:
            print(f"[質問] {question}\n")
            print(ask(question, history))
            question = input("\n続けて質問できます（終了するには空行）: ").strip()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
