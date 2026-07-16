import argparse
import os
import sys
import time
from pathlib import Path
from vision import GameVision
from solver import WordGameSolver
from printer_control import PrinterControl

SCRIPT_DIR = Path(__file__).resolve().parent
LOCK_PATH = SCRIPT_DIR / ".gamebot.lock"

def acquire_lock():
    # refuse to start if another instance's lock is still held by a live process
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)  # raises if no such process is running
            print(f"another game bot instance (pid {pid}) appears to be running; aborting.")
            print(f"if this is stale (e.g. after a crash), delete {LOCK_PATH} and try again.")
            sys.exit(1)
        except (ValueError, ProcessLookupError, OSError):
            pass  # stale lock file, safe to take over

    LOCK_PATH.write_text(str(os.getpid()))

def release_lock():
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass

def pause_for_phone_removal(control):
    control.set_status("Remove phone, then press Enter")
    print()
    print("OCR and solving are done. Remove the phone from the bed now --")
    print("the stylus will still move/tap/swipe through the solved coordinates,")
    print("just with nothing there to touch, so you can verify alignment safely.")
    input("press Enter once the phone is clear to continue...")

def play_word_hunt(vision, solver, control, pause_before_move=False):
    # capture, ocr, solve, swipe
    print("capturing screen for word hunt...")
    frame = vision.capture_frame()
    marker_centers = vision.detect_markers(frame)
    warped = vision.warp_phone_screen(frame, marker_centers)

    print("running ocr on letter grid...")
    grid, cell_centers = vision.extract_word_hunt_grid(warped)

    print("detected grid:")
    for row in grid:
        print("  " + " ".join(row))

    # flag unrecognized letters
    if any("?" in row for row in grid):
        print("warning: some letters were not recognized correctly. proceeding anyway.")

    print("solving word hunt...")
    words = solver.solve_word_hunt(grid)
    print(f"found {len(words)} possible words")

    if not words:
        print("no words found.")
        return

    # prepare printer
    control.set_status("Word Hunt: starting...")
    if not control.setup_printer():
        print("failed to prepare printer (see above). aborting.")
        return

    if pause_before_move:
        pause_for_phone_removal(control)

    # longest words first, for max score
    words_played = 0
    max_words = control.config.get("max_wordhunt_words", 35) # time limit ~30-35 moves

    for word, path in words:
        if words_played >= max_words:
            break

        print(f"playing word: {word.upper()} (length {len(word)})")
        control.set_status(f"Word Hunt: {word.upper()}")

        # grid path -> pixel coords
        pixel_path = []
        for r, c in path:
            pixel_path.append(cell_centers[(r, c)])

        if not control.swipe(pixel_path):
            print("swipe failed; stopping word hunt early so we don't compound the error.")
            break
        words_played += 1

        # buffer for the app to register the swipe (motion itself is already done)
        time.sleep(0.2)

    print(f"word hunt game finished. played {words_played} words.")

def play_anagrams(vision, solver, control, pause_before_move=False):
    # capture, ocr, solve, tap
    print("capturing screen for anagrams...")
    frame = vision.capture_frame()
    marker_centers = vision.detect_markers(frame)
    warped = vision.warp_phone_screen(frame, marker_centers)

    print("running ocr on letters...")
    letters, bubble_coords, submit_coord = vision.extract_anagrams_letters(warped)

    print(f"detected letters: {letters}")
    if "?" in letters:
        print("warning: some letters were not recognized correctly. proceeding anyway.")

    print("solving anagrams...")
    words = solver.solve_anagrams(letters)
    print(f"found {len(words)} possible words")

    if not words:
        print("no words found.")
        return

    # prepare printer
    control.set_status("Anagrams: starting...")
    if not control.setup_printer():
        print("failed to prepare printer (see above). aborting.")
        return

    if pause_before_move:
        pause_for_phone_removal(control)

    words_played = 0
    max_words = control.config.get("max_anagram_words", 40) # cap to fit the round timer
    for word in words:
        if words_played >= max_words:
            break

        print(f"playing word: {word.upper()}")
        control.set_status(f"Anagrams: {word.upper()}")

        # map word letters to bubbles, handling duplicates
        available_bubbles = list(enumerate(letters))
        tap_coordinates = []
        possible = True

        for char in word:
            found = False
            for idx, (b_idx, b_char) in enumerate(available_bubbles):
                if b_char == char:
                    tap_coordinates.append(bubble_coords[b_idx])
                    available_bubbles.pop(idx)
                    found = True
                    break
            if not found:
                possible = False
                break

        if not possible:
            print(f"skipping word {word} due to bubble matching error")
            continue

        tap_failed = False
        for px, py in tap_coordinates:
            if not control.tap(px, py):
                print("tap failed; stopping anagrams early so we don't compound the error.")
                tap_failed = True
                break
            time.sleep(0.1) # buffer for app to register the tap
        if tap_failed:
            break

        # submit button tap
        if not control.tap(submit_coord[0], submit_coord[1]):
            print("submit tap failed; stopping anagrams early so we don't compound the error.")
            break
        words_played += 1

        # buffer between words
        time.sleep(0.3)

    print(f"anagrams game finished. played {words_played} words.")

def main():
    # parse args, dispatch to game routine
    parser = argparse.ArgumentParser(description="voron gamepigeon bot player")
    parser.add_argument("--game", required=True, choices=["wordhunt", "anagrams"], help="game type to play")
    parser.add_argument("--dry-run", action="store_true", help="simulate play without moving printer")
    parser.add_argument("--pause-before-move", action="store_true",
                         help="run ocr/solve for real, then pause so you can remove the phone before the "
                              "printer executes the solved taps/swipes -- lets you verify coordinate "
                              "alignment in free air with nothing to damage")
    args = parser.parse_args()

    # resolve paths relative to script dir, not caller's cwd
    config_path = SCRIPT_DIR / "config.json"
    dictionary_path = SCRIPT_DIR / "dictionary.txt"

    print("loading configuration...")
    try:
        vision = GameVision(str(config_path))
        solver = WordGameSolver(str(dictionary_path))
        control = PrinterControl(str(config_path), dry_run=args.dry_run)
    except Exception as e:
        print(f"failed to initialize system: {e}")
        sys.exit(1)

    if not args.dry_run:
        acquire_lock()

    try:
        if args.game == "wordhunt":
            play_word_hunt(vision, solver, control, pause_before_move=args.pause_before_move)
        elif args.game == "anagrams":
            play_anagrams(vision, solver, control, pause_before_move=args.pause_before_move)
    finally:
        # always lift stylus, even on error
        control.park()
        if not args.dry_run:
            release_lock()

if __name__ == "__main__":
    main()
