import argparse
import sys

def filter_words(input_path, output_path, min_len, max_len):
    # read words from the source file, filter by length, and save to target file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            words = f.read().splitlines()
    except Exception as e:
        print(f"error reading input file: {e}")
        sys.exit(1)

    # filter words between min_len and max_len inclusive
    filtered = [w.strip() for w in words if min_len <= len(w.strip()) <= max_len]

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered) + "\n")
        print(f"successfully wrote {len(filtered)} words to {output_path}")
    except Exception as e:
        print(f"error writing output file: {e}")
        sys.exit(1)

def main():
    # parse arguments for input file, max length, min length, and output file
    parser = argparse.ArgumentParser(description="filter words in a text file by length range")
    parser.add_argument("input_file", help="path to the input text file of words")
    parser.add_argument("max_len", type=int, help="maximum length of words to keep")
    parser.add_argument("--min", "-m", type=int, default=0, help="minimum length of words to keep (default: 0)")
    parser.add_argument("--output", "-o", help="path to the output text file (defaults to filtered_words.txt)")
    
    args = parser.parse_args()

    # set default output file path if not specified
    output_file = args.output if args.output else "filtered_words.txt"
    
    filter_words(args.input_file, output_file, args.min, args.max_len)

if __name__ == "__main__":
    main()
