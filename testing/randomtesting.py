	

#!/usr/bin/env python3

import time
from pathlib import Path
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgbcontrol = pipeline.create(dai.node.XLinkIn)
#stillEnc = pipeline.create(dai.node.VideoEncoder)
xoutJpeg = pipeline.create(dai.node.XLinkOut)
#xoutRgb = pipeline.create(dai.node.XLinkOut)

camRgbcontrol.setStreamName("rgbcontrol")
xoutJpeg.setStreamName("jpeg")
#xoutRgb.setStreamName("rgb")

# Properties
camRgbcontrol.setMaxDataSize(1000)
camRgbcontrol.setNumFrames(1)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
#stillEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)


# Linking
#camRgb.video.link(xoutRgb.input)
camRgbcontrol.out.link(camRgb.inputControl)
camRgb.still.link(xoutJpeg.input)
#stillEnc.bitstream.link(xoutJpeg.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.setLogLevel(dai.LogLevel.DEBUG)
    device.setLogOutputLevel(dai.LogLevel.DEBUG)
    # Output queue will be used to get the rgb frames from the output defined above
    #qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    qControl = device.getInputQueue("rgbcontrol")
    qJpeg = device.getOutputQueue("jpeg")

    # Make sure the destination path is present before starting to store the examples
    dirName = "rgb_data"
    Path(dirName).mkdir(parents=True, exist_ok=True)
    print("starting main loop")
    while True:
        '''
        inRgb = qRgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise

        if inRgb is not None:
            cv2.imshow("rgb", inRgb.getCvFrame())
        '''
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        qControl.send(ctrl)
        time.sleep(1)

        encFrame = qJpeg.tryGet()
        if(encFrame != None):
            cv2.imwrite(f"{dirName}/{int(time.time() * 1000)}.jpeg", encFrame.getCvFrame())
            '''
            with open(f"{dirName}/{int(time.time() * 1000)}.jpeg", "wb") as f:
                f.write(bytearray(encFrame.getData()))
                f.close()
            '''

        if cv2.waitKey(1) == ord('q'):
            break
