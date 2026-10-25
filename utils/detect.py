from __future__ import annotations
import sys
from ultralytics import YOLO
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = Path("weights")
MODEL_PATH.mkdir(exist_ok=True)
OUTPUT_PATH = "detected.png"

class DetectionModel:
    def __init__(self) -> None:
        log.info("Try loading detection model..")
        try:
            self.model = YOLO(MODEL_PATH / "yolo11n.pt")
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            log.error(traceback.format_exc())
            
    
    def detect(self) -> None:
        log.info("Trying to inference and detect the objects...")
        try:
            
            results = self.model(
                sys.argv[1],
                classes=[2,5,7],
                conf=0.25,
                agnostic_nms=True
            )
            
            r = results[0]
            h, w = r.orig_shape
            print(f"image size: {w}x{h}")
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                name = r.names[int(box.cls)]
                conf = float(box.conf)
                area_pct = (x2- x1) * (y2 - y1) / (w * h) * 100
                print(f"{name:6s} conf={conf:.2f} box=({x1:.0f},{y1:.0f} - ({x2:.0f}, {y2:.0f})   area={area_pct:.1f}%")
                
            r.save(filename=OUTPUT_PATH)
            return
        except Exception as e:
            log.error(f"Unexpected error detecting objects: {str(e)}")
            log.error(traceback.format_exc())
    
    
    def get_model(self):
        return self.model        
            



detector = DetectionModel()



# if __name__ == "__main__":
#     detector = DetectionModel()
#     detector.detect()