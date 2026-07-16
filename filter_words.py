import argparse
import sys

def load_wordset(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())

def filter_words(input_path, output_path, min_len, max_len, reference_path=None):
    # read words from the source file, filter by length (and optionally against
    # a reference wordlist), and save to target file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            words = f.read().splitlines()
    except Exception as e:
        print(f"error reading input file: {e}")
        sys.exit(1)

    reference = None
    if reference_path:
        try:
            reference = load_wordset(reference_path)
            print(f"loaded {len(reference)} words from reference list {reference_path}")
        except Exception as e:
            print(f"error reading reference file: {e}")
            sys.exit(1)

    # filter words between min_len and max_len inclusive, and against the
    # reference wordlist if given -- this weeds out scrabble-only junk
    # (abbreviations, obscure inflections) that a real word bank won't accept
    filtered = []
    for w in words:
        w = w.strip().lower()
        if not (min_len <= len(w) <= max_len):
            continue
        if reference is not None and w not in reference:
            continue
        filtered.append(w)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered) + "\n")
        print(f"successfully wrote {len(filtered)} words to {output_path}")
    except Exception as e:
        print(f"error writing output file: {e}")
        sys.exit(1)

def main():
    # parse arguments for input file, max length, min length, output file, and reference wordlist
    parser = argparse.ArgumentParser(description="filter words in a text file by length range and an optional reference wordlist")
    parser.add_argument("input_file", help="path to the input text file of words")
    parser.add_argument("max_len", type=int, help="maximum length of words to keep")
    parser.add_argument("--min", "-m", type=int, default=0, help="minimum length of words to keep (default: 0)")
    parser.add_argument("--output", "-o", help="path to the output text file (defaults to filtered_words.txt)")
    parser.add_argument("--reference", "-r",
                         help="path to a reference wordlist (e.g. /usr/share/dict/words) -- only words that "
                              "also appear in this list are kept, filtering out scrabble-only junk that a "
                              "smaller word bank like GamePigeon's won't accept")

    args = parser.parse_args()

    # set default output file path if not specified
    output_file = args.output if args.output else "filtered_words.txt"

    filter_words(args.input_file, output_file, args.min, args.max_len, args.reference)

if __name__ == "__main__":
    main()
