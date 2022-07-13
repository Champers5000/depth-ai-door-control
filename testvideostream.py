import cv2
import os
import depthai as dai
from flask import Flask, render_template, Response, request
from thread_trace import thread_trace
app = Flask(__name__)



# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
camRgb.setPreviewSize(4208, 3120)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)
inRgbcvFrame = None

def gen_frames():  # generate frame by frame from camera
    global inRgbcvFrame
    while True:
        # Capture frame-by-frame
        if inRgbcvFrame.all() != None:
            frame = inRgbcvFrame  # read the camera frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

t_app = thread_trace(target=app.run)


# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    t_app.start()

    framecount =0
    while True:
        inRgbcvFrame = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("bgr", inRgbcvFrame)

        if t_app.is_alive() == False:
            t_app.start()
        
        k = cv2.waitKey(0)

        if k == ord('q'):
            t_app.kill()
            t_app.join()
            break

        if k == ord('c'):
            cv2.imwrite("~/Downloads/antispoof-facerecognition/"+str(framecount)+".png", inRgbcvFrame)
            print("writing image")
            

        framecount+=1
            