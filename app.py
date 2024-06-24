from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import tempfile

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
print(f"Model loaded: {model}")

# Define the path to save annotated images
save_path = os.path.join(os.path.dirname(__file__), 'annotated_images')
if not os.path.exists(save_path):
    os.makedirs(save_path)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    algorithm = request.form.get('algorithm', 'yolov8')
    file_type = file.content_type

    if algorithm == 'yolov8':
        model = YOLO('yolov8n.pt')
    elif algorithm == 'faster_rcnn':
        return jsonify({'error': 'Faster R-CNN not implemented yet'}), 501
    else:
        return jsonify({'error': 'Unsupported algorithm selected'}), 400

    if 'image' in file_type:
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print(f"Image shape: {img.shape}")
        results = model(img)
        print(f"Number of total detections: {len(results[0].boxes)}")
        print(f"Number of human detections: {sum(1 for box in results[0].boxes if box.cls == 0)}")
        annotated_img, annotated_image_path = annotate_image(file.filename, img, results)
        precision, recall, confidence_scores, TP, FP, FN = calculate_metrics(results, img.shape)
        print(f'Precision: {precision}, Recall: {recall}, Confidence scores: {confidence_scores}')
        img_str = encode_image(annotated_img)
        return jsonify({
            'detections': results_to_dict(results),
            'annotated_image': img_str,
            'image_path': annotated_image_path,
            'precision': precision,
            'recall': recall,
            'confidence_scores': confidence_scores,
            'true_positive': TP,
            'false_positive': FP,
            'false_negative': FN
        })
    elif 'video' in file_type:
        # Process video
        with tempfile.TemporaryDirectory() as tempdir:
            video_path = os.path.join(tempdir, file.filename)
            file.save(video_path)
            frames = process_video(video_path, model)
            annotated_video_path = os.path.join(save_path, f'annotated_{os.path.basename(file.filename)}')
            save_annotated_video(frames, annotated_video_path)
            precision, recall, confidence_scores, TP, FP, FN = calculate_metrics_for_video(frames, model)
            return jsonify({
                'annotated_video': annotated_video_path,
                'precision': precision,
                'recall': recall,
                'confidence_scores': confidence_scores,
                'true_positive': TP,
                'false_positive': FP,
                'false_negative': FN
            })
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame, _ = annotate_image(os.path.basename(video_path), frame, results)
        frames.append(annotated_frame)
    cap.release()
    return frames

def annotate_image(filename, image, results):
    annotated_img = image.copy()
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Class ID 0 is for humans
                bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
                cv2.rectangle(annotated_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    annotated_image_path = os.path.join(save_path, f'annotated_{filename}')
    cv2.imwrite(annotated_image_path, annotated_img)
    return annotated_img, annotated_image_path

def results_to_dict(results):
    all_detections = []
    for result in results:
        detections = []
        for box in result.boxes:
            if box.cls == 0:  # Class ID 0 is for humans
                bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
                confidence = box.conf.item()
                detections.append({'bbox': bbox, 'confidence': confidence})
        all_detections.append(detections)
    return all_detections

def encode_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    img_str = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode('utf-8')
    return img_str

def save_annotated_video(frames, output_path):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def calculate_metrics(results, img_shape):
    ground_truth_boxes = get_ground_truth_boxes(img_shape)

    TP, FP, FN = 0, 0, 0
    detected_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Only consider human detections
                detected_boxes.append(box.xyxy[0].tolist())

    print(f"Number of human detections: {len(detected_boxes)}")
    print(f"Detected human boxes: {detected_boxes}")

    matched_gt_boxes = []
    for gt_box in ground_truth_boxes:
        matched = False
        for det_box in detected_boxes:
            iou = calculate_iou(gt_box, det_box)
            if iou > 0.1:  # Adjust IoU threshold as needed
                TP += 1
                matched_gt_boxes.append(gt_box)
                matched = True
                print(f"Matched: GT {gt_box}, Det {det_box}, IoU {iou}")
                break
        if not matched:
            FN += 1
            print(f"Unmatched GT: {gt_box}")

    FP = len(detected_boxes) - TP

    print(f'TP: {TP}, FP: {FP}, FN: {FN}')

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    confidence_scores = [box.conf.item() for result in results for box in result.boxes if box.cls == 0]

    return precision, recall, confidence_scores, TP, FP, FN

def get_ground_truth_boxes(img_shape):
    # Adjust these values based on actual human positions in your test images
    return [
        [int(img_shape[1] * 0.2), int(img_shape[0] * 0.6), int(img_shape[1] * 0.3), int(img_shape[0] * 0.9)],
        [int(img_shape[1] * 0.3), int(img_shape[0] * 0.6), int(img_shape[1] * 0.4), int(img_shape[0] * 0.9)],
        [int(img_shape[1] * 0.4), int(img_shape[0] * 0.6), int(img_shape[1] * 0.5), int(img_shape[0] * 0.9)],
        [int(img_shape[1] * 0.5), int(img_shape[0] * 0.6), int(img_shape[1] * 0.6), int(img_shape[0] * 0.9)],
        [int(img_shape[1] * 0.6), int(img_shape[0] * 0.6), int(img_shape[1] * 0.7), int(img_shape[0] * 0.9)]
    ]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
def calculate_metrics_for_video(frames, model):
    TP_total, FP_total, FN_total = 0, 0, 0
    precision_total, recall_total = 0, 0
    all_confidence_scores = []

    for frame in frames:
        results = model(frame)
        detected_boxes = []
        frame_confidence_scores = []

        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Only consider human detections
                    detected_boxes.append(box.xyxy[0].tolist())
                    frame_confidence_scores.append(box.conf.item())

        ground_truth_boxes = get_ground_truth_boxes(frame.shape)
        TP, FP, FN = 0, 0, 0

        matched_gt_boxes = []
        for gt_box in ground_truth_boxes:
            matched = False
            for det_box in detected_boxes:
                iou = calculate_iou(gt_box, det_box)
                if iou > 0.1:  # Adjust IoU threshold as needed
                    TP += 1
                    matched_gt_boxes.append(gt_box)
                    matched = True
                    break
            if not matched:
                FN += 1

        FP = len(detected_boxes) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        TP_total += TP
        FP_total += FP
        FN_total += FN
        precision_total += precision
        recall_total += recall

        all_confidence_scores.extend(frame_confidence_scores)

    num_frames = len(frames)
    if num_frames > 0:
        avg_precision = precision_total / num_frames
        avg_recall = recall_total / num_frames
    else:
        avg_precision, avg_recall = 0, 0

    return avg_precision, avg_recall, all_confidence_scores, TP_total, FP_total, FN_total



if __name__ == '__main__':
    app.run(debug=True)