from flask import Flask, request, jsonify

from deepface import DeepFace

import cv2

import os


app = Flask(__name__)


# Path to the Database directory

# Path to the Database directory

DB_PATH = os.path.expanduser('~/hrm/Database')

def faceRecognition(image_path, emp_name):

    model = DeepFace.find(img_path=image_path, db_path="./Database", enforce_detection=False, model_name="VGG-Face", threshold=0.9)

    prediction = str(model[0]["identity"][0])

    parts = prediction.split('/')

    name = parts[-2]

    print(name)

    if emp_name == str(name):

        return True

    else:

        return False
    
@app.route('/recognize', methods=['POST'])

def recognize():

    if 'image' not in request.files or 'id' not in request.form:

        return jsonify({"error": "Please provide an image and employee name"}), 400


    image = request.files['image']

    emp_name = request.form['id']


    image_path = f"./temp/{image.filename}"

    image.save(image_path)


    try:

        result = faceRecognition(image_path, emp_name)

        return jsonify({"recognized": result})

    except Exception as e:

        return jsonify({"error": str(e)}), 500
    
@app.route('/extract_frames', methods=['POST'])

def extract_frames():

    if 'video' not in request.files or 'id' not in request.form:

        return jsonify({"error": "Please provide a video file and employee id"}), 400


    video = request.files['video']

    emp_name = request.form['id']


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


@app.route('/check_employee', methods=['GET'])

def check_employee():

    employee_name = request.args.get('id')

    if not employee_name:

        return jsonify({"error": "Please provide an employee name"}), 400


    # Check if the employee directory exists

    employee_path = os.path.join(DB_PATH, employee_name)

    if os.path.exists(employee_path) and os.path.isdir(employee_path):

        return jsonify({"exists": True})

    else:

        return jsonify({"exists": False})

    


@app.route('/list_employees', methods=['GET'])

def list_employees():

    # List all directories in the Database directory

    employee_names = [name for name in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, name))]

    return jsonify({"employees": employee_names})





if __name__ == '__main__':

    os.makedirs('./temp', exist_ok=True)

    os.makedirs('./Database', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True)
    
    