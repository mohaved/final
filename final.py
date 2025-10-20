# flask beye3mel page el web
from flask import Flask, render_template, Response, send_from_directory, jsonify, request
# cv2 eli betsha8al el camera compyter vision
import cv2
# bete3amel ma3 el arrays
import numpy as np
# betnazam el files w betgib mawa3id el sowar
import os
import serial
import time
# 3lshan el multitasking
from threading import Lock
# beysha8al el server
app = Flask(__name__)
# beysha8al el camera
camera = cv2.VideoCapture(1)
# bey2alel el ta25ir fi el frames
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
# law el lon el a7mar zahar
red_detected = False
red_frame = None
#law mesh me7aded el amaken
roi = None  
# 3lshan law esta5demt kaza thread
lock = Lock()

moni = serial.Serial("COM3", 9600, timeout= 1)
# bete3mel dir el gallery 
GALLERY_DIR = os.path.join('static', 'gallery')
#bet make sure en el dir mawgoud 3lshan law mesh mawgoud te3melo
os.makedirs(GALLERY_DIR, exist_ok=True)
# route el file beta3 el html
@app.route('/')
def index():
    return render_template('index.html')
# page el gallery
@app.route('/gallery')
def gallery():
# betezher el sowar el a5ira el awal
    images = sorted(os.listdir(GALLERY_DIR), reverse=True)
# gowa file el html beyezherha
    return render_template('gallery.html', images=images)
# el route beta3 el live feed
@app.route('/video_feed')
# heya generate frames eli fe3leyan betezher el frames
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# deih eli bet7arak el roi men mkanha beta5od input lel 4 corner betou3 el moraba3 fi pixels
@app.route('/update_roi', methods=['POST'])
def update_roi():
    global roi
    data = request.get_json()
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    w = int(data.get('w', 0))
    h = int(data.get('h', 0))
    with lock:
        roi = [x, y, w, h]
    return jsonify({'status': 'ROI updated'})
# lma a3ouz araga3 el camera t detect el lon el a7mar fi kol mkan 3ady mesh roi
@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    global roi
    with lock:
        roi = None
    return jsonify({'status': 'ROI cleared'})
# range el a7mar fi el HSV
def generate_frames():
    global red_detected, red_frame, roi

#       H      0 dah el lon el a7mar 
#       S      255 dah ya3ni el lon naki awy
#       V      255 deih ya3ni el lon fi aksa setou3o
#                          H   S    V 
    lower_red1 = np.array([0, 120, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 80])
    upper_red2 = np.array([179, 255, 255])

# beynadaf el soura kernel dah
    kernel = np.ones((5,5), np.uint8)
# bet read kol frame law ma3arafetsh betetla3 men el loop
    while True:
        success, frame = camera.read()
        if not success:
            break

        with lock:
            current_roi = roi.copy() if roi else None

# law fi roi beteshta8al gowah bas
        if current_roi:
            rx, ry, rw, rh = current_roi
# beyet2aked en el ab3ad mazbouta fi el roi
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, frame.shape[1] - rx)
            rh = min(rh, frame.shape[0] - ry)

            roi_frame = frame[ry:ry+rh, rx:rx+rw]
# bet7awel el BGR eli howa (BLUE,GREEN,RED) L hsv(HUE,SATURATION,VALUE)
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

# bete3mel mask
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
# beyshil el no2at el so8ayara
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
# betdawar 3la el manate2 el 7amra w betgib 7ododha
            contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 5000:
                    if not red_detected:
                        red_detected = True
                        # حفظ الصورة الكاملة لما يحصل كشف أول مرة
                        timestamp = int(time.time())
                        filename = f"gallery_{timestamp}.jpg"
                        filepath = os.path.join(GALLERY_DIR, filename)
                        moni.write(b'1')
                        cv2.imwrite(filepath, frame)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (rx + x, ry + y), (rx + x + w, ry + y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "Red Object (ROI)", (rx + x, ry + y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    red_detected = False
                    moni.write(b'0')

            else:
                red_detected = False

            # رسم مستطيل ROI
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        else:
            # لو مفيش ROI، نكشف على الصورة كلها (زي الأول)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2

            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 3000:
                    if not red_detected:
                        red_detected = True
                        timestamp = int(time.time())
                        filename = f"gallery_{timestamp}.jpg"
                        filepath = os.path.join(GALLERY_DIR, filename)
                        cv2.imwrite(filepath, frame)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(frame, "Red Object", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    red_detected = False
            else:
                red_detected = False

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/clear_gallery', methods=['POST'])
def clear_gallery():
    for file in os.listdir(GALLERY_DIR):
        os.remove(os.path.join(GALLERY_DIR, file))
    return jsonify({'status': 'Gallery cleared'})

if __name__ == '__main__':
    app.run(debug=False)
