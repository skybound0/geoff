import cv2
import sys
from pathlib import Path
from vision import GameVision
from printer_control import PrinterControl

SCRIPT_DIR = Path(__file__).resolve().parent

def run_calibration():
    # diagnostic capture: camera, markers, coordinate mapping
    config_path = str(SCRIPT_DIR / "config.json")
    print("initializing vision system...")
    try:
        vision = GameVision(config_path)
    except Exception as e:
        print(f"failed to initialize vision system: {e}")
        sys.exit(1)
        
    print("capturing camera frame...")
    try:
        frame = vision.capture_frame()
    except Exception as e:
        print(f"failed to capture camera frame: {e}")
        sys.exit(1)
        
    raw_frame_path = str(SCRIPT_DIR / "raw_frame.jpg")
    cv2.imwrite(raw_frame_path, frame)
    print(f"saved raw camera capture to {raw_frame_path}")
    
    print("detecting aruco markers...")
    marker_centers = vision.detect_markers(frame)
    
    if not marker_centers:
        print("no aruco markers detected. please check your camera view, lighting, and marker IDs.")
        sys.exit(1)
        
    print(f"detected markers: {list(marker_centers.keys())}")
    for marker_id, center in marker_centers.items():
        print(f"  marker ID {marker_id}: pixel coordinate {center}")
        
    missing = [i for i in [0, 1, 2, 3] if i not in marker_centers]
    if missing:
        print(f"missing markers: {missing}. all 4 markers (0, 1, 2, 3) must be visible to warp screen.")
        sys.exit(1)
        
    print("all 4 markers detected successfully! warping screen view...")
    try:
        warped = vision.warp_phone_screen(frame, marker_centers)
        warped_screen_path = str(SCRIPT_DIR / "warped_screen.jpg")
        cv2.imwrite(warped_screen_path, warped)
        print(f"saved warped phone screen to {warped_screen_path}")
    except Exception as e:
        print(f"failed to warp screen: {e}")
        sys.exit(1)
        
    print("checking coordinate mapping...")
    try:
        control = PrinterControl(config_path, dry_run=True)
        if control.bed_bounds:
            print(f"printer bed bounds (from moonraker): {control.bed_bounds}")
        else:
            print("warning: could not determine printer bed bounds; out-of-bounds moves will not be caught during play")
        print("verifying warped-pixel-to-printer coordinates:")
        test_points = [(0, 0), (1000, 0), (1000, 1000), (0, 1000), (500, 500)]
        for px, py in test_points:
            x, y = control.pixel_to_printer(px, py)
            print(f"  warped pixel ({px}, {py}) -> printer physical ({x:.2f} mm, {y:.2f} mm)")
    except Exception as e:
        print(f"failed to check printer coordinate transformation: {e}")
        sys.exit(1)
        
    print("calibration test completed successfully.")

if __name__ == "__main__":
    run_calibration()
