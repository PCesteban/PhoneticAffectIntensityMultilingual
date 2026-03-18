import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.embedding import Embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Spanish phoneme embeddings.")
    parser.add_argument("--corpus-source", type=str, default="wikipedia", choices=["legacy", "wikipedia"])
    parser.add_argument("--word-max-samples", type=int, default=20000)
    parser.add_argument("--part-max-samples", type=int, default=5000)
    parser.add_argument(
        "--verbosity",
        type=str,
        default="summary",
        choices=["quiet", "summary", "full"],
        help="Logging verbosity for corpus/vector generation.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate embedding plots.")
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="Compute token similarities and save them under data/output.",
    )
    parser.add_argument("--similarity-topn", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    em = Embedding(
        lang="es",
        word_max_samples=args.word_max_samples,
        part_max_samples=args.part_max_samples,
        verbosity=args.verbosity,
        corpus_source=args.corpus_source,
    )
    model_name = "phoneme_embedding_wiki" if args.corpus_source == "wikipedia" else "phoneme_embedding"
    em.part_embedding(model_name=model_name, size=150, syllable=False)

    if args.plot:
        model_file_stem = "{0}_es".format(model_name)
        em.plot(model_file_stem)
        em.plot_clusters(model_file_stem)

    if args.similarity:
        em.get_similarity(
            model_name=model_name,
            topn=args.similarity_topn,
            verbosity=args.verbosity,
            save_output=True,
        )


if __name__ == "__main__":
    main()
