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
