from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import supervision as sv
from ultralytics import YOLOv10
import cv2
import os
from urllib.parse import quote
import subprocess

app = Flask(__name__)

# YOLO 모델 로드
model = YOLOv10('C:/Users/Owner/Desktop/DL_Web_Video/yolov10n.pt')

# 업로드 폴더 설정
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def reencode_video(input_path, output_path):
    ffmpeg_path = 'C:/Users/Owner/AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe'  
    try:
        command = [
            ffmpeg_path,
            '-i', input_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-movflags', 'faststart',
            output_path
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during re-encoding: {str(e)}")
        return False
    except FileNotFoundError as e:
        print(f"FFmpeg not found: {str(e)}")
        return False
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
        
        if file:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)

            video_path = filename

            frames = []
            cap = cv2.VideoCapture(video_path)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    break

            cnt = 0
            frame_lst = []
            for frame in frames:
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                label_annotator = sv.LabelAnnotator()
                annotated_image = bounding_box_annotator.annotate(
                    scene=frame, detections=detections)
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections)
                final_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                frame_lst.append(final_image)
                cnt+=1
                
            width = final_image.shape[1]
            height = final_image.shape[0]
            fps = cnt/20

            output_filename = 'output_' + file.filename.rsplit('.', 1)[0] + '.mp4'
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            output_path = output_path.replace("\\", "/")
            temp_output_path = output_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            for frame in frame_lst:
                out.write(frame)
            out.release()

            # 동영상 파일 재인코딩
            if not reencode_video(temp_output_path, output_path):
                return jsonify({'error': '동영상 파일 재인코딩 중 오류가 발생했습니다.'}), 500

            # 재인코딩된 파일이 제대로 생성되었는지 확인
            if not os.path.exists(output_path):
                return jsonify({'error': '동영상 파일이 생성되지 않았습니다.'}), 500

            # 임시 파일 삭제
            os.remove(temp_output_path)

            # 상대 경로 반환
            relative_path = os.path.join(OUTPUT_FOLDER, output_filename).replace("\\", "/")
            return jsonify({'video_path': '/outputs/' + quote(output_filename)}), 200

    except Exception as e:
        print(f"Error: {str(e)}")  # 예외 메시지를 콘솔에 출력
        return jsonify({'error': '오류가 발생했습니다. 다시 시도해주세요.'}), 500

@app.route('/outputs/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)