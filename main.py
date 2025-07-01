# let's read a barcode like the machine (product scanner in super market )

# using packages 
# pip install opencv-python 
# pip install pydub 
# pip install pyzbar 

# opencv-python
# pyzbar
# brew install zbar
# pip3 install pydub
# ffmpeg


import cv2 
from pyzbar.pyzbar import decode
from pydub import AudioSegment
from pydub.playback import play

# Load the beep sound (MP3 file)
beep_sound = AudioSegment.from_file("beep.mp3", format="mp3")

# Capture webcam 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Flip the image like a mirror image
    frame = cv2.flip(frame, 1)

    # Detect the barcode
    detectedBarcode = decode(frame) if frame is not None else []

    # If no barcode is detected
    if not detectedBarcode:
        print("No Barcode Detected")

    # If barcode is detected
    else:
        # Process detected barcodes
        for barcode in detectedBarcode:
            if barcode.data:
                # Play the beep sound
                play(beep_sound)

                # Display the barcode data on the frame
                cv2.putText(frame, str(barcode.data.decode('utf-8')), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
                cv2.imwrite("code.png", frame)

    # Display the frame
    cv2.imshow('scanner', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()