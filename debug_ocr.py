import argparse
import sys
from pathlib import Path

import cv2

from vision import GameVision

SCRIPT_DIR = Path(__file__).resolve().parent


def debug_wordhunt(vision, warped):
    # mirrors the crop geometry in vision.extract_word_hunt_grid
    grid_conf = vision.config.get("wordhunt_grid", {
        "x_start": 80, "x_end": 920, "y_start": 350, "y_end": 850, "crop_padding": 0.25
    })
    x_start, x_end = grid_conf["x_start"], grid_conf["x_end"]
    y_start, y_end = grid_conf["y_start"], grid_conf["y_end"]
    crop_padding = grid_conf.get("crop_padding", 0.25)
    cell_w = (x_end - x_start) // 4
    cell_h = (y_end - y_start) // 4

    grid, _ = vision.extract_word_hunt_grid(warped)

    annotated = warped.copy()
    for r in range(4):
        for c in range(4):
            x1 = x_start + c * cell_w
            y1 = y_start + r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            pad_w, pad_h = int(cell_w * crop_padding), int(cell_h * crop_padding)
            # yellow = full cell, green = actual region fed to ocr
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.rectangle(annotated, (x1 + pad_w, y1 + pad_h), (x2 - pad_w, y2 - pad_h), (0, 255, 0), 2)
            cv2.putText(annotated, grid[r][c].upper(), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return annotated, grid


def debug_anagrams(vision, warped):
    # mirrors the crop geometry in vision.extract_anagrams_letters
    anagram_conf = vision.config.get("anagrams", {
        "num_letters": 6, "x_start": 283, "x_end": 718,
        "y_start": 871, "y_end": 963, "crop_padding": 0.15,
        "submit_x": 500, "submit_y": 631
    })
    num_letters = anagram_conf["num_letters"]
    x_start, x_end = anagram_conf["x_start"], anagram_conf["x_end"]
    y_start, y_end = anagram_conf["y_start"], anagram_conf["y_end"]
    crop_padding = anagram_conf.get("crop_padding", 0.15)
    box_w = (x_end - x_start) / num_letters
    pad_w = int(box_w * crop_padding)
    pad_h = int((y_end - y_start) * crop_padding)

    letters, bubble_coords, submit_coord = vision.extract_anagrams_letters(warped)

    annotated = warped.copy()
    for i, letter in enumerate(letters):
        x1 = int(x_start + i * box_w)
        x2 = int(x_start + (i + 1) * box_w)
        # yellow = full tile, green = actual region fed to ocr
        cv2.rectangle(annotated, (x1, y_start), (x2, y_end), (0, 255, 255), 1)
        cv2.rectangle(annotated, (x1 + pad_w, y_start + pad_h), (x2 - pad_w, y_end - pad_h), (0, 255, 0), 2)
        cv2.putText(annotated, letter.upper(), (x1 + 5, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(annotated, submit_coord, 8, (255, 0, 0), 2)
    return annotated, letters


def main():
    parser = argparse.ArgumentParser(description="visualize ocr crop regions against a captured phone screen")
    parser.add_argument("--game", required=True, choices=["wordhunt", "anagrams"])
    parser.add_argument("--reuse-warped", action="store_true",
                         help="reuse warped_screen.jpg from a previous calibrate.py run instead of capturing a new frame")
    args = parser.parse_args()

    config_path = str(SCRIPT_DIR / "config.json")
    vision = GameVision(config_path)

    warped_path = SCRIPT_DIR / "warped_screen.jpg"
    if args.reuse_warped:
        if not warped_path.exists():
            print(f"{warped_path} not found; run calibrate.py first or omit --reuse-warped")
            sys.exit(1)
        warped = cv2.imread(str(warped_path))
    else:
        frame = vision.capture_frame()
        marker_centers = vision.detect_markers(frame)
        warped = vision.warp_phone_screen(frame, marker_centers)

    if args.game == "wordhunt":
        annotated, detected = debug_wordhunt(vision, warped)
    else:
        annotated, detected = debug_anagrams(vision, warped)

    out_path = SCRIPT_DIR / f"debug_{args.game}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"detected: {detected}")
    print(f"saved annotated crop overlay to {out_path}")
    print("compare the green boxes against the actual letters in the image -- "
          "if a box is off-center on a letter, empty, or catching part of a neighboring letter, "
          "adjust the wordhunt_grid/anagrams geometry in config.json")


if __name__ == "__main__":
    main()
