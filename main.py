import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam-resolution", nargs=2, default=[480, 640], type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)

    zone_polygon = (ZONE_POLYGON * args.webcam_resolution).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=1,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        results = model(frame, agnostic_nms=True)
        print(results)

        detections = sv.Detections.from_yolov8(results[0])
        detections = detections[detections.class_id != 0]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(30) == 27:
            print(frame.shape)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
