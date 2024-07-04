from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import os

app = Flask(__name__)

def faceRecognition(image_path, emp_name):
    model = DeepFace.find(img_path=image_path, db_path="./Database", enforce_detection=False, model_name="VGG-Face", threshold=0.9)
    print(model[0]["identity"][0][0].split('/')[1])
    prediction = model[0]['identity'][0].split('/')[1].split('\\')[1]
    name = prediction.split(' ')
    if emp_name == str(name[0] + " " + name[1]):
        return True
    else:
        return False

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files or 'emp_name' not in request.form:
        return jsonify({"error": "Please provide an image and employee name"}), 400

    image = request.files['image']
    emp_name = request.form['emp_name']

    image_path = f"./temp/{image.filename}"
    image.save(image_path)

    try:
        result = faceRecognition(image_path, emp_name)
        return jsonify({"recognized": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
########################################
def video_to_images(video_path, emp_name):
    num_frames = 10  # Fixed number of frames to extract
    output_dir = f"./Database/{emp_name}/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": f"Cannot open video file {video_path}"}
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    while frame_count < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    return {"message": f"Finished extracting frames. {frame_count} frames saved in {output_dir}"}

@app.route('/extract_frames', methods=['POST'])
def extract_frames():
    if 'video' not in request.files or 'emp_name' not in request.form:
        return jsonify({"error": "Please provide a video file and employee name"}), 400

    video = request.files['video']
    emp_name = request.form['emp_name']

    video_path = f"./temp/{video.filename}"
    video.save(video_path)

    try:
        result = video_to_images(video_path, emp_name)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    os.makedirs('./temp', exist_ok=True)
    os.makedirs('./Database', exist_ok=True)
    app.run(debug=True)
