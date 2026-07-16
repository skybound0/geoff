import cv2
import numpy as np
import json
import os

# set tesseract command path if defined in config
try:
    import pytesseract
except ImportError:
    pytesseract = None

class GameVision:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # configure tesseract binary path if specified and exists
        if pytesseract and "tesseract_cmd" in self.config:
            if os.path.exists(self.config["tesseract_cmd"]):
                pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]

        # set up camera
        self.camera_url = self.config.get("camera_url", 0)
        # handle integers for direct camera devices
        if isinstance(self.camera_url, str) and self.camera_url.isdigit():
            self.camera_url = int(self.camera_url)

        # build detector once, not per call
        self.aruco_detector, self.aruco_new_api = self._build_aruco_detector()

    def capture_frame(self):
        # open, grab one frame, close
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            raise IOError(f"could not open camera source: {self.camera_url}")
        
        # read a few frames to let camera auto-exposure settle
        for _ in range(5):
            ret, frame = cap.read()
        
        cap.release()
        if not ret:
            raise IOError("failed to retrieve frame from camera")
        return frame

    def _build_aruco_detector(self):
        # old/new opencv aruco api compat
        try:
            # new opencv api (4.7.0+)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            return detector, True
        except AttributeError:
            # old opencv api
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters_create()
            return (dictionary, parameters), False

    def detect_markers(self, frame):
        # detects 4x4_50 aruco markers in the frame
        if self.aruco_new_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        else:
            dictionary, parameters = self.aruco_detector
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            
        marker_centers = {}
        if ids is not None:
            # flatten ids list
            flat_ids = ids.flatten()
            for corner_set, marker_id in zip(corners, flat_ids):
                # corner_set has shape (1, 4, 2)
                pts = corner_set[0]
                # calculate center of the marker
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                marker_centers[int(marker_id)] = (center_x, center_y)
                
        return marker_centers

    def warp_phone_screen(self, frame, marker_centers):
        # warp phone screen to normalized 1000x1000
        required_ids = [0, 1, 2, 3]
        for req_id in required_ids:
            if req_id not in marker_centers:
                raise ValueError(f"cannot warp screen, missing marker ID {req_id}")
                
        src_pts = np.array([
            marker_centers[0],
            marker_centers[1],
            marker_centers[2],
            marker_centers[3]
        ], dtype=np.float32)
        
        # normalized screen destination points (1000x1000)
        dst_pts = np.array([
            [0, 0],
            [1000, 0],
            [1000, 1000],
            [0, 1000]
        ], dtype=np.float32)
        
        h_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, h_matrix, (1000, 1000))
        return warped

    def ocr_letter(self, img_crop):
        # ocr a single letter crop
        if not pytesseract:
            raise ImportError("pytesseract is not installed")
            
        # preprocess image for ocr: grayscale, resize, threshold
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_CUBIC)
        
        # apply thresholding to get sharp black-on-white text
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # invert if light-on-dark; tesseract wants black-on-white
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)
        if white_pixels < black_pixels:
            thresh = cv2.bitwise_not(thresh)
            
        # single-char whitelist config -- lowercase candidates are kept even though
        # gamepigeon only renders uppercase, because tesseract's lstm engine can
        # refuse to output anything (rather than fall back to the next-best
        # in-whitelist letter) when its top guess for a glyph is excluded outright;
        # e.g. a thin vertical stroke's top guess is often lowercase "l", and without
        # it in the whitelist that comes back empty instead of as a correctable "l"
        custom_config = r"--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # clean the extracted character (case matters here -- lowercase "l" vs
        # uppercase "I" is exactly the ambiguity corrections below fixes, so
        # check corrections before uppercasing, not after)
        char = text.strip()
        if not char:
            return "?"

        # handle common ocr misinterpretations
        corrections = {
            "0": "O", "1": "I", "|": "I", "l": "I", "5": "S", "8": "B", "2": "Z", "6": "G"
        }
        return corrections.get(char[0], char[0]).upper()

    def extract_word_hunt_grid(self, warped_screen):
        # extract 4x4 letter grid from warped screen
        grid_conf = self.config.get("wordhunt_grid", {
            "x_start": 80, "x_end": 920, "y_start": 350, "y_end": 850, "crop_padding": 0.25
        })
        grid_x_start = grid_conf["x_start"]
        grid_x_end = grid_conf["x_end"]
        grid_y_start = grid_conf["y_start"]
        grid_y_end = grid_conf["y_end"]
        crop_padding = grid_conf.get("crop_padding", 0.25)
        
        cell_w = (grid_x_end - grid_x_start) // 4
        cell_h = (grid_y_end - grid_y_start) // 4
        
        grid = []
        # cell center coords in warped pixels
        cell_centers = {}
        
        for r in range(4):
            grid_row = []
            for c in range(4):
                # calculate bounding box of cell
                x1 = grid_x_start + c * cell_w
                y1 = grid_y_start + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                # shrink crop to stay inside the tile
                pad_w = int(cell_w * crop_padding)
                pad_h = int(cell_h * crop_padding)
                crop = warped_screen[y1+pad_h:y2-pad_h, x1+pad_w:x2-pad_w]
                
                # recognize letter
                letter = self.ocr_letter(crop).lower()
                grid_row.append(letter)
                
                # center in warped space
                cx = x1 + cell_w // 2
                cy = y1 + cell_h // 2
                cell_centers[(r, c)] = (cx, cy)
                
            grid.append(grid_row)
            
        return grid, cell_centers

    def extract_anagrams_letters(self, warped_screen):
        # extract letters arranged in a horizontal row of tiles
        anagram_conf = self.config.get("anagrams", {
            "num_letters": 6, "x_start": 283, "x_end": 718,
            "y_start": 871, "y_end": 963, "crop_padding": 0.15,
            "submit_x": 500, "submit_y": 631
        })
        num_letters = anagram_conf["num_letters"]
        x_start = anagram_conf["x_start"]
        x_end = anagram_conf["x_end"]
        y_start = anagram_conf["y_start"]
        y_end = anagram_conf["y_end"]
        crop_padding = anagram_conf.get("crop_padding", 0.15)

        box_w = (x_end - x_start) / num_letters
        pad_w = int(box_w * crop_padding)
        pad_h = int((y_end - y_start) * crop_padding)

        letters = []
        bubble_coords = []

        for i in range(num_letters):
            x1 = int(x_start + i * box_w)
            x2 = int(x_start + (i + 1) * box_w)

            # shrink crop to stay inside the tile
            crop = warped_screen[y_start+pad_h:y_end-pad_h, x1+pad_w:x2-pad_w]

            letter = self.ocr_letter(crop).lower()
            letters.append(letter)

            # tile center in warped space
            cx = (x1 + x2) // 2
            cy = (y_start + y_end) // 2
            bubble_coords.append((cx, cy))

        submit_coord = (anagram_conf["submit_x"], anagram_conf["submit_y"])

        return letters, bubble_coords, submit_coord
