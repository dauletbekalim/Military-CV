
from ultralytics import YOLO
import cv2
import json


model = YOLO("yolov8n.pt")

ARMOR_CLASSES = {"car", "truck", "bus"}   
TROOP_CLASSES = {"person"}

def analyze_image(image_path: str, output_json: str = None):
    results = model.predict(image_path, conf=0.4)  
    image = cv2.imread(image_path)

    counts = {"armored_vehicles": 0, "troops": 0}
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = r.names[cls_id]

            if cls_name in ARMOR_CLASSES:
                category = "armored_vehicle"
                counts["armored_vehicles"] += 1
            elif cls_name in TROOP_CLASSES:
                category = "troop"
                counts["troops"] += 1
            else:
                continue 

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, category, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({
                "category": category,
                "location": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            })


    annotated_path = image_path.replace(".jpg", "_annotated.jpg")
    cv2.imwrite(annotated_path, image)

    result = {"counts": counts, "detections": detections, "annotated_image": annotated_path}

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"[DEMO] Found {counts['armored_vehicles']} armored vehicles and {counts['troops']} troops")
    print(f"[DEMO] Annotated image saved to {annotated_path}")
    return result


if __name__ == "__main__":
    analyze_image("sample1.jpg", "detections.json")
