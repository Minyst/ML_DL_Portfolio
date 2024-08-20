from flask import Flask, request, jsonify, render_template
import os
import supervision as sv
from ultralytics import YOLOv10
import cv2

app = Flask(__name__)

# YOLO 모델 로드
model = YOLOv10('C:/Users/Owner/Desktop/DL_Web_Image/yolov10n.pt')

# 클래스 이름 정의
class_names = ['Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal', 'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire']

# 업로드 폴더 설정
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'}), 400
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 이미지 로드
        image = cv2.imread(file_path)
        
        # YOLO 모델을 사용하여 예측
        results = model(image)
        detections = sv.Detections.from_ultralytics(results[0])

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        # 결과 이미지 저장
        annotated_image_path = os.path.join(UPLOAD_FOLDER, 'annotated_' + file.filename)
        cv2.imwrite(annotated_image_path, annotated_image)

        # JSON 형식으로 변환
        detections_json = results[0].tojson(normalize=False)

        return jsonify({
            'detections': detections_json,
            'image_path': annotated_image_path
        })

if __name__ == '__main__':
    app.run(debug=True)
