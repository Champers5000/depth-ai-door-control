logpics = False

import cv2
import numpy as np
import depthai as dai
import os
from face_auth import enroll_face, delist_face, authenticate_emb, init, removenames, takePicture
import blobconverter
import time
from flask import Flask, render_template, Response, request
from thread_trace import thread_trace
from threading import Lock
app = Flask(__name__)

# Define Depth Classification model input size
DEPTH_NN_INPUT_SIZE = (64, 64)

# Define Face Detection model name and input size
# If you define the blob make sure the DET_MODEL_NAME and DET_ZOO_TYsPE are None
DET_INPUT_SIZE = (300,300)
DET_MODEL_NAME = "face-detection-retail-0004"
DET_ZOO_TYPE = "depthai"
#det_blob_path = "data/depth-classification-models/face-detection-retail-0004.blob"

# Define Face Recognition model name and input size
# If you define the blob make sure the REC_MODEL_NAME and REC_ZOO_TYPE are None
REC_MODEL_NAME = "Sphereface"
REC_ZOO_TYPE = "intel"
#rec_blob_path = "data/depth-classification-models/Sphereface.blob"

frame_count = 0  # Frame count
fps = 0  # Placeholder fps value
prev_frame_time = 0  # Used to record the time when we processed last frames
new_frame_time = 0  # Used to record the time at which we processed current frames

# Load image of a lock in locked position
locked_img = cv2.imread(os.path.join('data', 'images', 'lock_grey.png'), -1)
# Load image of a lock in unlocked position
unlocked_img = cv2.imread(os.path.join('data', 'images', 'lock_open_grey.png'), -1)

#global variables
name = "unnamed"
userin = ""
quitthisloop = False
frame = None
prediction = ""

# Set status colors
status_color = {
    'Authenticated': (0, 255, 0),
    'Unauthenticated': (0, 0, 255),
    'Spoof Detected': (0, 0, 255),
    'No Face Detected': (0, 0, 255)
}

#methods needed to stream the video
def gen_frames():  # generate frame by frame from camera
    global frame
    while True:
        # Capture frame-by-frame
        if frame.all() != None:
            outputframe = frame  # read the camera frame
            ret, buffer = cv2.imencode('.jpg', outputframe)
            outputframe = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + outputframe + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def startflask():
    app.run(host='0.0.0.0')

#methods needed for depthai
# Create DepthAi pipeline
def create_depthai_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()


    #Create color camera pipeline here
    # Define sources and outputs
    rgbControl = pipeline.create(dai.node.XLinkIn)
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutJpeg = pipeline.create(dai.node.XLinkOut)
    # stream names
    rgbControl.setStreamName("rgbcontrol")
    xoutJpeg.setStreamName("jpg")
    # Properties
    rgbControl.setMaxDataSize(1000)
    rgbControl.setNumFrames(1)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
    # Linking
    rgbControl.out.link(camRgb.inputControl)
    camRgb.still.link(xoutJpeg.input)


    # Define a source - two mono (grayscale) cameras
    left = pipeline.createMonoCamera()
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.createMonoCamera()
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create a node that will produce the depth map
    depth = pipeline.createStereoDepth()
    depth.setConfidenceThreshold(200)
    depth.setOutputRectified(True)  # The rectified streams are horizontally mirrored by default
    depth.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    depth.setExtendedDisparity(True)  # For better close range depth perception

    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7  # For depth filtering
    depth.setMedianFilter(median)

    # Linking mono cameras with depth node
    left.out.link(depth.left)
    right.out.link(depth.right)

    # Create left output
    xOutRight = pipeline.createXLinkOut()
    xOutRight.setStreamName("right")
    depth.rectifiedRight.link(xOutRight.input)

    # Create depth output
    xOutDisp = pipeline.createXLinkOut()
    xOutDisp.setStreamName("disparity")
    depth.disparity.link(xOutDisp.input)

    # Create input and output node for Depth Classification
    xDepthIn = pipeline.createXLinkIn()
    xDepthIn.setStreamName("depth_in")
    xOutDepthNn = pipeline.createXLinkOut()
    xOutDepthNn.setStreamName("depth_nn")

    # Define Depth Classification NN node
    depthNn = pipeline.createNeuralNetwork()
    depthNn.setBlobPath("data/depth-classification-models/depth_classification_ipscaled_model.blob")
    depthNn.input.setBlocking(False)

    # Linking
    xDepthIn.out.link(depthNn.input)
    depthNn.out.link(xOutDepthNn.input)

    # Convert detection model from OMZ to blob
    if DET_MODEL_NAME is not None:
        facedet_blob_path = blobconverter.from_zoo(
            name=DET_MODEL_NAME,
            shaves=6,
            zoo_type=DET_ZOO_TYPE
        )

    # Create Face Detection NN node
    faceDetNn = pipeline.createMobileNetDetectionNetwork()
    faceDetNn.setConfidenceThreshold(0.75)
    faceDetNn.setBlobPath(facedet_blob_path)

    # Create ImageManip to convert grayscale mono camera frame to RGB
    copyManip = pipeline.createImageManip()
    depth.rectifiedRight.link(copyManip.inputImage)
    # copyManip.initialConfig.setHorizontalFlip(True)
    copyManip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    # Create ImageManip to preprocess input frame for detection NN
    detManip = pipeline.createImageManip()
    # detManip.initialConfig.setHorizontalFlip(True)
    detManip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
    detManip.initialConfig.setKeepAspectRatio(False)

    # Linking detection ImageManip to detection NN
    copyManip.out.link(detManip.inputImage)
    detManip.out.link(faceDetNn.input)

    # Create output steam for detection output
    xOutDet = pipeline.createXLinkOut()
    xOutDet.setStreamName('det_out')
    faceDetNn.out.link(xOutDet.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to crop the initial frame for recognition NN
    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)
    script.setScriptPath("script.py")

    # Set inputs for script node
    copyManip.out.link(script.inputs['frame'])
    faceDetNn.out.link(script.inputs['face_det_in'])

    # Convert recognition model from OMZ to blob
    if REC_MODEL_NAME is not None:
        facerec_blob_path = blobconverter.from_zoo(
            name=REC_MODEL_NAME,
            shaves=6,
            zoo_type=REC_ZOO_TYPE
        )

    # Create Face Recognition NN node
    faceRecNn = pipeline.createNeuralNetwork()
    faceRecNn.setBlobPath(facerec_blob_path)

    # Create ImageManip to preprocess frame for recognition NN
    recManip = pipeline.createImageManip()

    # Set recognition ImageManipConfig from script node
    script.outputs['manip_cfg'].link(recManip.inputConfig)
    script.outputs['manip_img'].link(recManip.inputImage)

    # Create output steam for recognition output
    xOutRec = pipeline.createXLinkOut()
    xOutRec.setStreamName('rec_out')
    faceRecNn.out.link(xOutRec.input)

    recManip.out.link(faceRecNn.input)

    return pipeline


# Overlay lock/unlock symbol on the frame
def overlay_symbol(frame, img, pos=(65, 100)):
    """
    This function overlays the image of lock/unlock
    if the authentication of the input frame
    is successful/failed.
    """
    # Offset value for the image of the lock/unlock
    symbol_x_offset = pos[0]
    symbol_y_offset = pos[1]

    # Find top left and bottom right coordinates
    # where to place the lock/unlock image
    y1, y2 = symbol_y_offset, symbol_y_offset + img.shape[0]
    x1, x2 = symbol_x_offset, symbol_x_offset + img.shape[1]

    # Scale down alpha channel between 0 and 1
    mask = img[:, :, 3]/255.0
    # Inverse of the alpha mask
    inv_mask = 1-mask

    # Iterate over the 3 channels - R, G and B
    for c in range(0, 3):
        # Add the lock/unlock image to the frame
        frame[y1:y2, x1:x2, c] = (mask * img[:, :, c] +
                                  inv_mask * frame[y1:y2, x1:x2, c])


# Display info on the frame
def display_info(frame, bbox, status, status_color, fps):
    global prediction
    # Display bounding box
    cv2.rectangle(frame, bbox, status_color[status], 2)

    # If spoof detected
    if status == 'Spoof Detected':
        # Display "Spoof detected" status on the bbox
        cv2.putText(frame, prediction, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Create background for showing details
    #cv2.rectangle(frame, (5, 5, 175, 150), (50, 0, 0), -1)

    # Display authentication status on the frame
    #cv2.putText(frame, status, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Display lock symbol
    #if status == 'Authenticated':
    #    overlay_symbol(frame, unlocked_img)
    #else:
    #    overlay_symbol(frame, locked_img)

    # Display instructions on the frame
    #cv2.putText(frame, 'Press E to Enroll Face.', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    #cv2.putText(frame, 'Press D to Delist Face.', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    #cv2.putText(frame, 'Press Q to Quit.', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

#main method run by thread
def main(device):
    global frame_count
    global fps
    global prev_frame_time
    global new_frame_time
    global name
    global userin
    global quitthisloop
    global frame
    global prediction
    # Start pipeline
    device.startPipeline()

    # Output queue to get the right camera frames
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    # Output queue to get the disparity map
    qDepth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    # Input queue to send face depth map to the device
    qDepthIn = device.getInputQueue(name="depth_in")

    # Output queue to get Depth Classification nn data
    qDepthNn = device.getOutputQueue(name="depth_nn", maxSize=4, blocking=False)

    # Output queue to get Face Recognition nn data
    qRec = device.getOutputQueue(name="rec_out", maxSize=4, blocking=False)

    # Output queue to get Face Detection nn data
    qDet = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)

    # Output queue for color camera
    qJpeg = device.getOutputQueue(name="jpg", maxSize=1, blocking=False)

    # Input for rgb camera to take pictures
    qControl = device.getInputQueue(name = "rgbcontrol")
    
    print("Starting main thread")
    while True:

        # Get right camera frame
        inRight = qRight.get()
        r_frame = inRight.getFrame()
        #r_frame = cv2.flip(r_frame, flipCode=1)

        # Get depth frame
        inDepth = qDepth.get()  # blocking call, will wait until a new data has arrived
        depth_frame = inDepth.getFrame()
        depth_frame = cv2.flip(depth_frame, flipCode=1)
        depth_frame = np.ascontiguousarray(depth_frame)
        inverted_depth_frame = depth_frame
        depth_frame = cv2.bitwise_not(depth_frame)

        # Apply color map to highlight the disparity info
        #depth_frame_cmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        # Show disparity frame
        #cv2.imshow("disparity", depth_frame_cmap)

        # Convert grayscale image frame to 'bgr' (opencv format)
        frame = cv2.cvtColor(r_frame, cv2.COLOR_GRAY2BGR)

        # Get image frame dimensions
        img_h, img_w = frame.shape[0:2]

        bbox = None
        mirroredbbox = None

        # Get detection NN output
        inDet = qDet.tryGet()

        if inDet is not None:
            # Get face bbox detections
            detections = inDet.detections

            if len(detections) != 0:
                # Use first detected face bbox
                detection = detections[0]
                # print(detection.confidence)
                x = int(detection.xmin * img_w)
                y = int(detection.ymin * img_h)
                w = int(detection.xmax * img_w - detection.xmin * img_w)
                h = int(detection.ymax * img_h - detection.ymin * img_h)
                bbox = (x, y, w, h)
                mirroredbbox = (img_w-x-w, y, w, h)

        face_embedding = None
        authenticated = False

        # Check if a face was detected in the frame
        if bbox:
            showthis = inverted_depth_frame[max(0, mirroredbbox[1]):mirroredbbox[1] + mirroredbbox[3], max(0, mirroredbbox[0]):mirroredbbox[0] + mirroredbbox[2]]
            average=showthis.sum()
            average /= (bbox[2]*bbox[3])
            showthis = cv2.applyColorMap(showthis, cv2.COLORMAP_JET)

            # Get face roi depth frame
            face_d = depth_frame[max(0, mirroredbbox[1]):mirroredbbox[1] + mirroredbbox[3], max(0, mirroredbbox[0]):mirroredbbox[0] + mirroredbbox[2]]
            #cv2.imshow("face_roi", face_d)

            # Preprocess face depth map for classification
            resized_face_d = cv2.resize(face_d, DEPTH_NN_INPUT_SIZE)
            resized_face_d = resized_face_d.astype('float16')

            # Create Depthai Imageframe
            img = dai.ImgFrame()
            img.setFrame(resized_face_d)
            img.setWidth(DEPTH_NN_INPUT_SIZE[0])
            img.setHeight(DEPTH_NN_INPUT_SIZE[1])
            img.setType(dai.ImgFrame.Type.GRAYF16)

            # Send face depth map to depthai pipeline for classification
            qDepthIn.send(img)

            # Get Depth Classification NN output
            inDepthNn = qDepthNn.tryGet()

            is_real = None
            if average>24:
                if inDepthNn is not None:
                    # Get prediction
                    cnn_output = inDepthNn.getLayerFp16("dense_2/Sigmoid")
                    #print(cnn_output[0])
                    if cnn_output[0] > .05:
                        prediction = 'spoofed'
                        is_real = False
                    else:
                        prediction = 'real'
                        is_real = True
                    #print(prediction)
            else:
                prediction = 'too close'

            if is_real:
                # Check if the face in the frame was authenticated

                # Get recognition NN output
                inRec = qRec.tryGet()
                if inRec is not None:
                    # Get embedding of the face
                    face_embedding = inRec.getFirstLayerFp16()
                    # print(len(face_embedding))

                    authenticated = authenticate_emb(face_embedding)

                if authenticated:
                    # Authenticated
                    status = 'Authenticated'
                    #send command to take a colored picture for logging
                    if logpics:
                        qControl.send(dai.CameraControl().setCaptureStill(True))

                else:
                    # Unauthenticated
                    status = 'Unauthenticated'
            else:
                # Spoof detected
                status = 'Spoof Detected'
        else:
            # No face detected
            status = 'No Face Detected'

        # Calculate average fps
        if frame_count % 10 == 0:
            # Time when we finish processing last 10 frames
            new_frame_time = time.time()

            # Fps will be number of frame processed in one second
            fps = 1 / ((new_frame_time - prev_frame_time)/10)
            prev_frame_time = new_frame_time
        
        # Display info on frame
        display_info(frame, bbox, status, status_color, fps)

        # Capture the key pressed
        key_pressed = cv2.waitKey(1) & 0xff

        lock.acquire()
        
        if(userin == 'i'):
            name = input("Input a new name >>> ")
        elif(userin == 'e'):
            if status == 'No Face Detected' or status == 'Spoof Detected':
                print("no face found to enroll")
            elif is_real == True:
                print("Enrolling face "+ name)
                enroll_face([face_embedding], name)
        elif(userin == 'd'):
            if is_real == True:
                delist_face([face_embedding])
        elif(userin == 'r'):
            removethisguy = input("Enter a name to delist. Enter 0 to delete all saved faces >>> ")
            removenames(removethisguy)
        if quitthisloop:
            print("Exiting main loop")
            lock.release()
            return
        userin = ""
        lock.release()
        
        #cv2.imshow("Authentication Cam", frame)

        if(status == "Authenticated" and logpics):
            takePicture(qJpeg, frame)
        
        # Increment frame count
        frame_count += 1



'''
        # Enrol the face if e was pressed
        if key_pressed == ord('e'):
            if is_real:
                print("enrolling face")
                enroll_face([face_embedding], name)
        # Delist the face if d was pressed
        elif key_pressed == ord('d'):
            if is_real:
                delist_face([face_embedding])
        elif key_pressed == ord('i'):
            print()
        	#somehow get inputexit()
        # Stop the program if q was pressed
        elif key_pressed == ord('q'):
            break

        # Display the final frame
        cv2.imshow("Authentication Cam", frame)
'''


def keyin():
    global userin
    global quitthisloop
    while True:
        if(userin == ""):
            tempuserin = input("Ready for input \n")
            lock.acquire()
            userin = tempuserin
            if(userin == "q"):
                quitthisloop = True
                lock.release()
                return
            lock.release()



#Start a different thread to load in all the faces
t_init = thread_trace(target=init)
t_init.start()

# Create Pipeline
pipeline = create_depthai_pipeline()
t_init.join()

t_app = thread_trace(target=startflask)
t_app.start()
# Initialize device and start Pipeline
with dai.Device(pipeline) as device:

    lock = Lock()
    #t_main = Thread(target=main, args=(device,))

    t_keyboardin = thread_trace(target = keyin)

    #t_main.start()
    t_keyboardin.start()
    main(device)

# Close all output windows
t_app.kill()
t_app.join()
exit(0)