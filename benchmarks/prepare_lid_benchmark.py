from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
}

SAMPLE_TEXTS = [
    "This is a realistic English sentence for language identification throughput.",
    "Ceci est une phrase francaise utilisee pour mesurer les performances du modele.",
    "Dies ist ein deutscher Beispielsatz fuer einen reproduzierbaren Benchmark.",
    "Esta es una frase en espanol para probar la inferencia por lotes.",
    "Questa e una frase italiana per valutare la velocita di predizione.",
    "To jest polskie zdanie testowe do pomiaru przepustowosci klasyfikacji.",
    "Este e um texto em portugues usado para benchmark de identificacao de idioma.",
    "Dit is een Nederlandse voorbeeldzin voor een taalherkenningsbenchmark.",
    "Dette er en norsk setning som brukes i en ytelsestest.",
    "Toto je ceska veta pro testovani vykonu rozpoznavani jazyka.",
    "Aceasta este o propozitie in limba romana pentru testarea modelului.",
    "Bu cumle Turkce dil tanima performansini olcmek icin kullanilir.",
    "Eto russkoe predlozhenie dlya proverki skorosti klassifikacii yazyka.",
    "これは言語識別の性能を測定するための日本語の文章です。",
    "这是一个用于测试语言识别吞吐量的中文句子。",
    "이 문장은 언어 식별 처리량을 측정하기 위한 한국어 예시입니다.",
    "هذه جملة عربية لاختبار سرعة التعرف على اللغة.",
    "यह भाषा पहचान प्रदर्शन को मापने के लिए एक हिंदी वाक्य है।",
]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")

    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    temporary.replace(destination)


def _write_input(path: Path, repeat: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index in range(repeat):
            for text in SAMPLE_TEXTS:
                output.write(f"{text} Document number {index}.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache") / "fasttext-parallel" / "lid",
    )
    parser.add_argument("--model-kind", choices=sorted(MODEL_URLS), default="bin")
    parser.add_argument("--repeat", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than zero")

    model_path = args.output_dir / f"lid.176.{args.model_kind}"
    input_path = args.output_dir / "lid_input.txt"

    if args.force or not model_path.exists():
        print(f"Downloading {MODEL_URLS[args.model_kind]} -> {model_path}")
        _download(MODEL_URLS[args.model_kind], model_path)
    else:
        print(f"Using existing model: {model_path}")

    if args.force or not input_path.exists():
        print(f"Writing multilingual input corpus -> {input_path}")
        _write_input(input_path, args.repeat)
    else:
        print(f"Using existing input: {input_path}")

    docs = args.repeat * len(SAMPLE_TEXTS)
    print(f"model={model_path}", file=sys.stderr)
    print(f"input={input_path}", file=sys.stderr)
    print(f"docs={docs}", file=sys.stderr)


if __name__ == "__main__":
    main()
