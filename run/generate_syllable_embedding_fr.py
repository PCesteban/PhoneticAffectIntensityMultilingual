import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.embedding import Embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Generate French syllable embeddings.")
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
        lang="fr",
        word_max_samples=args.word_max_samples,
        part_max_samples=args.part_max_samples,
        verbosity=args.verbosity,
    )
    em.part_embedding(model_name="syllable_embedding", size=150)

    if args.plot:
        em.plot("syllable_embedding_fr")
        em.plot_clusters("syllable_embedding_fr")

    if args.similarity:
        em.get_similarity(
            model_name="syllable_embedding",
            topn=args.similarity_topn,
            verbosity=args.verbosity,
            save_output=True,
        )


if __name__ == "__main__":
    main()
