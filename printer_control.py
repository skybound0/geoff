import json
import requests
import numpy as np
import cv2

class PrinterControl:
    def __init__(self, config_path="config.json", dry_run=False):
        self.dry_run = dry_run
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.moonraker_url = self.config.get("moonraker_url", "http://localhost:7125")
        self.z_up = self.config.get("z_up", 15.0)
        self.z_down = self.config.get("z_down", 5.0)
        self.f_travel = self.config.get("f_travel", 6000.0)
        self.f_swipe = self.config.get("f_swipe", 2000.0)
        self.f_z = self.config.get("f_z", 3000.0)
        self.touch_dwell_ms = self.config.get("touch_dwell_ms", 100)

        # reuse one connection instead of reconnecting for every gcode call
        self.session = requests.Session()

        # warped pixels (1000x1000) -> printer mm transform
        self.init_coordinate_transform()

        # real axis limits, to reject out-of-bounds moves
        self.bed_bounds = self.fetch_bed_bounds()

    def fetch_bed_bounds(self):
        # axis limits from moonraker (x/y only)
        try:
            url = f"{self.moonraker_url}/printer/objects/query?toolhead=axis_minimum,axis_maximum"
            response = self.session.get(url, timeout=5)
            result = response.json()["result"]["status"]["toolhead"]
            x_min, y_min = result["axis_minimum"][0], result["axis_minimum"][1]
            x_max, y_max = result["axis_maximum"][0], result["axis_maximum"][1]
            return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        except Exception as e:
            print(f"warning: could not fetch printer bed bounds from moonraker ({e}); skipping coordinate sanity checks")
            return None

    def in_bounds(self, x_mm, y_mm):
        # true if within bed bounds (or bounds unknown, so unchecked)
        if self.bed_bounds is None:
            return True
        return (self.bed_bounds["x_min"] <= x_mm <= self.bed_bounds["x_max"] and
                self.bed_bounds["y_min"] <= y_mm <= self.bed_bounds["y_max"])

    def check_homed(self):
        # x/y/z must be homed before trusting absolute moves
        if self.dry_run:
            return True
        try:
            url = f"{self.moonraker_url}/printer/objects/query?toolhead=homed_axes"
            response = self.session.get(url, timeout=5)
            homed = response.json()["result"]["status"]["toolhead"]["homed_axes"]
            return all(axis in homed for axis in "xyz")
        except Exception as e:
            print(f"failed to check homed status: {e}")
            return False

    def init_coordinate_transform(self):
        # source points: normalized corners of the warped image
        src_pts = np.array([
            [0.0, 0.0],       # top-left
            [1000.0, 0.0],     # top-right
            [1000.0, 1000.0],   # bottom-right
            [0.0, 1000.0]      # bottom-left
        ], dtype=np.float32)
        
        # destination points: physical printer coordinates from config
        aruco_config = self.config["aruco_markers"]
        dst_pts = np.array([
            [aruco_config["0"]["x"], aruco_config["0"]["y"]],
            [aruco_config["1"]["x"], aruco_config["1"]["y"]],
            [aruco_config["2"]["x"], aruco_config["2"]["y"]],
            [aruco_config["3"]["x"], aruco_config["3"]["y"]]
        ], dtype=np.float32)
        
        # warped pixels -> printer mm homography
        self.h_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def pixel_to_printer(self, px, py):
        # warped pixel -> printer mm
        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.h_matrix)
        x_mm = float(transformed[0][0][0])
        y_mm = float(transformed[0][0][1])
        return x_mm, y_mm

    def send_gcode(self, script):
        # send gcode block via moonraker
        if self.dry_run:
            print(f"[dry run] sending gcode:\n{script}\n")
            return True
            
        url = f"{self.moonraker_url}/printer/gcode/script"
        headers = {"Content-Type": "application/json"}
        payload = {"script": script}
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                return True
            else:
                print(f"error sending gcode to printer: {response.text}")
                return False
        except Exception as e:
            print(f"failed to connect to printer api: {e}")
            return False

    def set_status(self, message):
        # status line on the display; nobody sees this script's stdout otherwise
        return self.send_gcode(f"M117 {message}\n")

    def setup_printer(self):
        # absolute positioning + lift stylus
        if not self.check_homed():
            print("printer is not homed on x/y/z; refusing to move. please home the printer first.")
            return False

        setup_script = (
            f"G90\n"                        # absolute positioning
            f"G1 Z{self.z_up} F{self.f_z}\n" # lift stylus
            f"M400\n"                        # wait for the move to physically finish
        )
        return self.send_gcode(setup_script)

    def park(self):
        # lift stylus, move to the configured park position, reset status; safe to call after a failure too
        gcode = f"G1 Z{self.z_up} F{self.f_z}\n"

        park_pos = self.config.get("park_position")
        if park_pos:
            x, y, z = park_pos["x"], park_pos["y"], park_pos["z"]
            if self.in_bounds(x, y):
                gcode += f"G1 X{x} Y{y} F{self.f_travel}\n"
                gcode += f"G1 Z{z} F{self.f_z}\n"
            else:
                print(f"warning: configured park_position ({x}, {y}) is outside the printer's bed bounds; skipping park move")

        gcode += "M400\n"
        self.send_gcode(gcode)
        self.set_status("Game Bot Idle")

    def tap(self, px, py):
        # move, drop, dwell, lift
        x_mm, y_mm = self.pixel_to_printer(px, py)
        if not self.in_bounds(x_mm, y_mm):
            print(f"refusing to tap: computed coordinate ({x_mm:.2f}, {y_mm:.2f}) mm is outside the printer's bed bounds")
            return False

        gcode = (
            f"; tap at pixel {px}, {py} -> printer {x_mm:.2f}, {y_mm:.2f}\n"
            f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{self.f_travel}\n"
            f"G1 Z{self.z_down} F{self.f_z}\n"
            f"G4 P{self.touch_dwell_ms}\n" # dwell so the touchscreen registers contact
            f"G1 Z{self.z_up} F{self.f_z}\n"
            f"M400\n"                     # wait for the move to physically finish before we return
        )
        return self.send_gcode(gcode)

    def swipe(self, points):
        # lift, move to start, drop, drag through points, lift
        if not points:
            return True

        # validate every point first, so a bad one aborts the whole swipe
        mm_points = [self.pixel_to_printer(px, py) for px, py in points]
        for x_mm, y_mm in mm_points:
            if not self.in_bounds(x_mm, y_mm):
                print(f"refusing to swipe: computed coordinate ({x_mm:.2f}, {y_mm:.2f}) mm is outside the printer's bed bounds")
                return False

        start_x, start_y = mm_points[0]
        gcode = (
            f"; swipe path starting at pixel {points[0][0]}, {points[0][1]} -> printer {start_x:.2f}, {start_y:.2f}\n"
            f"G1 X{start_x:.2f} Y{start_y:.2f} F{self.f_travel}\n"
            f"G1 Z{self.z_down} F{self.f_z}\n"
        )

        for x_mm, y_mm in mm_points[1:]:
            gcode += f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{self.f_swipe}\n"

        gcode += f"G1 Z{self.z_up} F{self.f_z}\nM400\n"

        return self.send_gcode(gcode)
