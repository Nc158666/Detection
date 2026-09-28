"""
Assignment 3 - Object Detection

Independent solution using a pretrained Ultralytics YOLO model.
No teacher-provided SSD/TensorRT files are required.

Put these two images beside this file:
    image1.jpg
    image2.jpg

Install:
    python -m pip install ultralytics opencv-python

Run:
    python your-detection.py

The first run may download yolo11n.pt automatically.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
IMAGE_FILES = ["image1.jpg", "image2.jpg"]
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_DIR = Path("detection_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_one(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Cannot open {image_path}. Put the image beside this script."
        )

    h, w = image.shape[:2]
    result = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    annotated = image.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        left, top, right, bottom = box.xyxy[0].tolist()
        left = max(0, min(w - 1, round(left)))
        top = max(0, min(h - 1, round(top)))
        right = max(0, min(w - 1, round(right)))
        bottom = max(0, min(h - 1, round(bottom)))

        width = max(0, right - left)
        height = max(0, bottom - top)
        area = width * height
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        class_name = names.get(class_id, str(class_id))

        d = {
            "ClassID": class_id,
            "Class": class_name,
            "Confidence": confidence,
            "Left": left,
            "Top": top,
            "Right": right,
            "Bottom": bottom,
            "Width": width,
            "Height": height,
            "Area": area,
            "Center": (center_x, center_y),
        }
        detections.append(d)

        cv2.rectangle(
            annotated, (left, top), (right, bottom), (0, 255, 0), 2
        )
        label = f"{class_name} {confidence:.3f}"
        cv2.putText(
            annotated, label, (left, max(25, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )

    return annotated, detections


def print_results(image_name, detections):
    print("\n" + "=" * 70)
    print(f"IMAGE: {image_name}")
    print("=" * 70)

    if not detections:
        print("No objects detected above the confidence threshold.")
        return

    for i, d in enumerate(detections, 1):
        print(f"\nDetection {i}")
        print(f"ClassID:     {d['ClassID']}")
        print(f"Class:       {d['Class']}")
        print(f"Confidence:  {d['Confidence']:.6f}")
        print(f"Left:        {d['Left']}")
        print(f"Top:         {d['Top']}")
        print(f"Right:       {d['Right']}")
        print(f"Bottom:      {d['Bottom']}")
        print(f"Width:       {d['Width']}")
        print(f"Height:      {d['Height']}")
        print(f"Area:        {d['Area']}")
        print(f"Center:      ({d['Center'][0]:.1f}, {d['Center'][1]:.1f})")


def save_report(all_results):
    report = OUTPUT_DIR / "detection_results.txt"

    with open(report, "w", encoding="utf-8") as f:
        f.write("Assignment 3 - Object Detection Results\n")
        f.write("=" * 70 + "\n\n")

        for image_name, detections in all_results:
            f.write(f"IMAGE: {image_name}\n")
            f.write("-" * 70 + "\n")

            if not detections:
                f.write("No objects detected above the confidence threshold.\n\n")
                continue

            for i, d in enumerate(detections, 1):
                f.write(f"Detection {i}\n")
                f.write(f"ClassID:     {d['ClassID']}\n")
                f.write(f"Class:       {d['Class']}\n")
                f.write(f"Confidence:  {d['Confidence']:.6f}\n")
                f.write(f"Left:        {d['Left']}\n")
                f.write(f"Top:         {d['Top']}\n")
                f.write(f"Right:       {d['Right']}\n")
                f.write(f"Bottom:      {d['Bottom']}\n")
                f.write(f"Width:       {d['Width']}\n")
                f.write(f"Height:      {d['Height']}\n")
                f.write(f"Area:        {d['Area']}\n")
                f.write(
                    f"Center:      "
                    f"({d['Center'][0]:.1f}, {d['Center'][1]:.1f})\n\n"
                )

            f.write("\n")

    return report


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    all_results = []

    for index, image_name in enumerate(IMAGE_FILES, 1):
        image_path = Path(image_name)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing {image_name}. "
                "Put it in the same folder as your-detection.py."
            )

        annotated, detections = detect_one(model, image_path)

        output_image = OUTPUT_DIR / f"detected_image{index}.jpg"
        cv2.imwrite(str(output_image), annotated)

        print_results(image_name, detections)
        print(f"\nSaved annotated image: {output_image}")

        all_results.append((image_name, detections))

    report = save_report(all_results)

    print("\n" + "=" * 70)
    print("Detection completed.")
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Text report:    {report}")
    print("=" * 70)


if __name__ == "__main__":
    main()
