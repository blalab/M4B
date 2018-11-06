# MFA for Buildings (M4B)
Pairing QR Codes with Faces for the built environment


## Inspiration
How can we use new technologies to enhance the tenant experience without sacrificing security?

## What it does
MFA for Buildings (M4B) is a standalone piece of software that remembers visitors and provides a way for them to use QR Codes to access the building by pairing their face and name to the guest pass. 

Some facts about it:
- Every new face is assigned a unique ID
- Even if you reboot the system, every face will keep its unique ID 
- Unique IDs are shared between cameras. If one camera sees a face in the lobby, another camera will recognize the same face on the 3rd floor.
- M4B's Neural Network continuously improves the face vector by learning from different angles and light conditions.
- M4B is designed to plug to existing security systems. It plays the Authentication role (not the Authorization)
- M4B is able to authenticate with different degrees of MFA: Face Only, Face+QR, QR Only. 
- M4B runs on premises, no internet connection is required.
- You can make M4B forget a face by just deleting its encoding file.
- You can train M4B from still photos, recorded video or live video feed.
- M4B is able to differentiate from a building occupant and a visitor.
- M4B is capable to read from up to 15 types of QRCode and BarCodes (Thanks to Pyzbar).
- M4B pairs QR Codes to Face IDs.
- QR Codes are only be valid when the assigned face is present in front of the camera.
- When the guest presents the Guest Pass QR Code to the camera at the Security Desk, M4B pairs it to the guest name and face.
- The network will remember the guest's name and face for future visits.
- The system can fall back to simple QR Code authentication if Facial Recognition needs to be deactivated.
- The authentication logs, can be sent to an anomaly detection system for further analysis.


## How we built it

The system uses :
OpenCV for face detection, face recognition
Zbar for QR Code detection and decoding
Numpy for image processing
Imutils for video frame capture

The program is entirely written on Python 3. 
Special thanks and credits to Adrian Rosebrock for his awesome OpenCV tutorials and code examples. The face recognition, face detection, QR Code reading parts of this code are based on his work.

## What's next for MFA for Buildings

We'd like to Pilot M4B in as many buildings as possible and integrate other Authentication factors like Handwritting, Gestures and Labeling.

## HARDWARE

OSX

M4B runs at least on a MacBook Pro with the following Hardware:
- Processor 3.3 GHz
- Intel Code i7
- 16 GB RAM

Raspberry Pi

M4B runs on the Raspberry Pi 3



## INSTALLATION


You'll have to install the following in your computer:

- Python 3

- OpenCV 4

For Mac
https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

For Raspberry Pi
https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/

- VirtualEnv
- VirtualEnv Wrapper

- PYZBAR
$ pip install pyzbar

- Face Recognition
$ pip install face_recognition

## USAGE

To create a user pass:

$ python guest_booking.py

To run the main program which will identify faces and read QR Codes:

$ python main.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --framesize 1600 --videosource 0


To forget all the faces:

$ source nuke.sh

## GUEST PASS

Tenant passes live in /g_pass an have the following shape:

```{'_id': '', 'guest': 'John Doe', 'token': '717c1', 'date': '20181106', 'time': '', 'guest_type': 'tenant_guest', 'face_id': '', 'created': '20181106-171012', 'host': ''}```


If you want to integrate to a Guest Management System your script should insert the generated token string in the "token" field for M4B to pair the face along with the Guest name in the "guest" field and the pass date in the "date" field. 


