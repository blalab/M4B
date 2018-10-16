# Release 20181016

# [QR]QR Code  
# [RE]Face Recognition 
# [ST]Short Term Memory  
# [LT]Long Term Memory 
# [CO]Face Detection Confidence 
# [SI]Image Save to disk
# [PA]Reload past memories
# [ME]Metadata
# [NA]User Name replacement
# [LC]Learning Cap
# [QF] QR Code - Face matching 
# [GU] Guest register + Auto Load + Guest Matching
# [LG] Long Term Guest face memory


# USAGE
# python main.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --framesize 1000

# [QR,FA,RE] import the necessary packages
from imutils.video import VideoStream #[RE]
from pyzbar import pyzbar
from imutils.video import FPS #[RE]
import face_recognition #[RE]
import numpy as np 
import argparse #[RE]
import datetime
import imutils #[RE]
import time #[RE]
import cv2 #[RE]
import pickle #[RE]
import uuid #[LT]
import os #[LT]
import json #[ME]

# [QR,RE]
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--framesize", type=int, default=400,
    help="Size of the frame to be processed")

ap.add_argument("-v", "--videosource", type=int, default=0,
    help="Video feed source")

# [QR] construct the argument parser and parse the arguments
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
    help="path to output CSV file containing barcodes")

# [RE] construct the argument parser and parse the arguments
ap.add_argument("-s", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")


# [CO] construct the argument parse and parse the arguments
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")


# [SI] 
ap.add_argument("-d", "--detectionth", type=int, default=50,
    help="sets the detection threshold to capture an encoding")

ap.add_argument("-n", "--confidenceth", type=float, default=0.9,
    help="sets the confidence threshold to capture an encoding")

ap.add_argument("-k", "--capturecap", type=int, default=10,
    help="maximum amount of encodings per user")

# [GU,LG]

ap.add_argument("-g", "--newguestbufferth", type=int, default=10,
    help="maximum amount of encodings per user")



# [QR,RE]
args = vars(ap.parse_args())

# [CO] load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# [RE,PA,LC] load the known faces and embeddings
print("[INFO] loading encodings + face detector...")

data = {"encodings": [], "names": []}


for subdir, dirs, files in os.walk("u_encodings"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file


        if filepath.endswith(".pickle"):

            part_data = ''
            with open(filepath, "rb") as f: 
                try:  
                    part_data_pickle = f.read()  
                    part_data = pickle.loads(part_data_pickle) 

                    #Load content in memory
                    data["encodings"].append(part_data["encoding"])
                    data["names"].append(part_data["uuid"])
                except:
                    print('%s is corrupt. Skipping'%filepath)
                    pass    
                finally:
                    f.close()


# [ME] Load User Metadata
uuid2name = {}
uuid2qr = {}


for subdir, dirs, files in os.walk("u_meta"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".json"):


            m_data = ''
            with open(filepath, "r") as f: 
                try:  
                    m_data_json = f.read()  
                    m_data = json.loads(m_data_json) 

                    #print(g_data)

                    #Load registered guests in memory
                    if not m_data["face_id"] == '' and not m_data["user_name"] == '':
                        uuid2name[m_data["face_id"]] = m_data["user_name"] 
                    
                    if not m_data["face_id"] == '' and not m_data["qr_id"] == '':
                        uuid2qr[m_data["face_id"]] = m_data["qr_id"]

                except:
                    print('%s is corrupt. Skipping'%filepath)
                    pass    
                finally:
                    f.close()


#print(uuid2name)
print(uuid2qr)


# [ME] Load Guest Passes

uuid2guestqr = {}
uuid2guestname = {}
unregistered_guests = {}


for subdir, dirs, files in os.walk("g_pass"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".json"):
            
            # Open the file
            g_data = ''
            with open(filepath, "r") as f: 
                try:  
                    g_data_json = f.read()  
                    g_data = json.loads(g_data_json) 

                    #print(g_data)

                    #Load today's registered guests in memory
                    if not g_data["face_id"]=='' and not g_data["token"]=='' and g_data["date"]==time.strftime("%Y%m%d"):
                        uuid2guestqr[g_data["face_id"]] = g_data["token"] 
                        uuid2guestname[g_data["face_id"]] = g_data["guest"]

                    #Load today's un-registered guests in memory
                    if g_data["face_id"] == '' and not g_data["token"] == '' and g_data["date"]==time.strftime("%Y%m%d"):
                        unregistered_guests[g_data["token"]] = g_data["guest"] 

                except:
                    print('%s is corrupt. Skipping'%filepath)
                    pass    
                finally:
                    f.close()
            

print(uuid2guestqr)
#print(uuid2guestname)
#print(unregistered_guests)


# [RE] Initialize object for new faces and embeddings
udata = {"encodings": [], "names": []}

# [RE] load OpenCV's Haar cascade for face detection

detector = cv2.CascadeClassifier(args["cascade"])

# [QR,RE] initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=args["videosource"]).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# [RE] start the FPS counter
fps = FPS().start()

# [QR] open the output CSV file for writing and initialize the set of
# [QR] barcodes found thus far
csv = open(args["output"], "w")
found = set()

# [RE] Buffer to store last cycle encoding
uencodings = []
unames = []
encodings = []

#[GU] Variable to keep track of stability of guest registration candidates
new_guest_buffer = ('','',0)


# [QR,RE] loop over the frames from the video stream
while True:
    # [QR,RE] grab the frame from the threaded video stream and resize it to
    # have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=args["framesize"])

    # [QR] find the barcodes in the frame and decode each of the barcodes
    barcodes = pyzbar.decode(frame)

    # [RE] convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # [RE] detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # [RE] OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    

    # [RE] compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # [CO] grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # [CO] pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    fdetection_centroids = []

    # [CO] loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # calculate geometric centroid of box
        c = (float(startX+((endX-startX)/2)),float(startY+((endY-startY)/2)),confidence)
        fdetection_centroids.append(c)

 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (255, 255, 0), 2)
        cv2.putText(frame, text, (endX, endY),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    #print(fdetection_centroids)

    present_barcodes = []


    #[QR] loop over the detected barcodes
    for barcode in barcodes:

        # extract the bounding box location of the barcode and draw
        # the bounding box surrounding the barcode on the image
        (x1, y1, w1, h1) = barcode.rect
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it
        # on our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        present_barcodes.append(barcodeData)


        # draw the barcode data and barcode type on the image
        text1 = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(frame, text1, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # if the barcode text is currently not in our CSV file, write
        # the timestamp + barcode to disk and update the set
        if barcodeData not in found:
            csv.write("{},{}\n".format(datetime.datetime.now(),
                barcodeData))
            csv.flush()
            found.add(barcodeData)

    

    # [RE] loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings

        #print(data)
        # search in Long Term memory
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)

        #print(matches)

        #print(last_encodings)

        # search in Short Term memory
        umatches = face_recognition.compare_faces(udata["encodings"],
            encoding)

        #print(unknown_matches)


        # check if we have found a match in the long term memory
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            #print('match!  : %s'%name)

        # check if we have found a match in the short term memory
        elif True in umatches:

            
            # find the indexes of all unknown faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            umatchedIdxs = [j for (j, d) in enumerate(umatches) if d]
            ucounts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for j in umatchedIdxs:
                uname = udata["names"][j]
                ucounts[uname] = ucounts.get(uname, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(ucounts, key=ucounts.get)

            #print('Candidate match! : %s'%name)


        else:

            # [LT,RE] New Candidate

            name = uuid.uuid4().hex

            uencodings.append(encoding)
            unames.append(name)

            #print('New Candidate! : %s'%name)


        
        # update the list of names
        names.append(name)

    udata = {"encodings": uencodings, "names": unames}

    #print(unames)

    frame_height, frame_width, frame_channels = frame.shape

    num_faces = len(encodings)

    foot1 = "Recognized faces:%s"%num_faces
    cv2.putText(frame, foot1, (10, frame_height-20 ), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (200, 200, 200), 2)


    # [RE] loop over the recognized faces
    for ((top, right, bottom, left), name, encoding) in zip(boxes, names, encodings):

        
        # [*] draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15

        if name in uuid2name:
            displayname = uuid2name[name]
        elif name in uuid2guestname:
            displayname = uuid2guestname[name]
        else:
            displayname = name

        cv2.putText(frame, displayname, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)


        
        ###
        
        
        # [CO] Calculate centroid of Face Detection Confidence box
        rcentroid = (float(left+((right-left)/2)),float(bottom+((top-bottom)/2)),name)
        

        # [CO] Match Face Detection Confidence box 
        # with Face Recognition box using 
        # centroids proximity and area similarity

        captured = False

        for dcentroid in fdetection_centroids :
            c_distance = ((rcentroid[0]-dcentroid[0])**2)+((rcentroid[1]-dcentroid[1])**2)**0.5
            c_result = (c_distance,dcentroid[2],rcentroid[2])

            #print(c_result)

            
            # [SI,ME,LC] Save User data to disk if the capture conditions are met. 
            if c_result[0]<(args["framesize"]/args["detectionth"]) and c_result[1]>args["confidenceth"] :

                # [LC]Before saving, check if this user already reached the 
                # encodings cap. If yes, abort saving (#TOFIX: something that
                # allows keep learning without causing memory leak)
                if os.path.exists("u_encodings/%s"%name):
                    path, dirs, files = os.walk("u_encodings/%s"%name).__next__()
                    if len(files) > args["capturecap"]:
                        #print("ENCODING CAP REACHED for :%s"%name)
                        continue

                
                timestr = time.strftime("%Y%m%d-%H%M%S")

                #Save Metadata
                meta_data = {'_id':'','face_id':name,'qr_id':'','user_name':'','access_start':'',\
                    'access_end':'','user_pin':'','user_type':'','last_seen':timestr,'user_references':''}

                met_filename = 'u_meta/%s/meta.json'%(name)
            

                if not os.path.exists(met_filename):
                    os.makedirs(os.path.dirname(met_filename), exist_ok=True)
                    with open(met_filename, "w") as f:
                        try:
                            print(meta_data)
                            f.write(json.dumps(meta_data))
                            print("Metadata CAPTURED")
                        except:
                            pass
                        finally:
                            f.close()



                # Save Encoding
                candidate_data = {"encoding":encoding,"uuid":name}
                enc_filename = 'u_encodings/%s/enc_%s.pickle'%(name,timestr)
                os.makedirs(os.path.dirname(enc_filename), exist_ok=True)
                with open(enc_filename, "wb") as f:
                    try:
                        f.write(pickle.dumps(candidate_data))
                        print("Encoding CAPTURED")
                    except:
                        pass
                    finally:
                        f.close()

                # Save Frame
                fra_filename = 'u_frames/%s/fra_%s.png'%(name,timestr)
                os.makedirs(os.path.dirname(fra_filename), exist_ok=True)
                try:
                    cv2.imwrite(fra_filename,frame)
                    print("Frame CAPTURED")
                finally:
                    pass

                
                #print(c_result)
                captured = True
                continue

        #print("Matching END")


        
        if captured:
            cv2.putText(frame, 'LEARNING', (right, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 2)

        

        

        #[QF] Figure out it there is a QR Code - Face match
        #looking for name in the access lists

        #print(name)

        if name in uuid2guestqr:
            #print('in guest list')
            # guest list lookup
            if uuid2guestqr[name] in present_barcodes:
                qrmatch = "Access Granted : Guest"
            else:
                qrmatch = None
        elif name in uuid2qr:
            #print('in tenant list')
            # tenant list lookup
            if uuid2qr[name] in present_barcodes:
                qrmatch = "Access Granted : Tenant"
            else:
                qrmatch = None     
        else:
            qrmatch = None

            if len(unregistered_guests)==0:
                continue

            #If the face hasn't been captured in the permanent encodings, skip
            if not os.path.exists('u_encodings/%s'%name):
                continue

            #[GU] Not found in any access list.
            # Match an unknown face with an unmatched barcode without a face

            if len(present_barcodes)==1 and num_faces==1:
                if present_barcodes[0] in unregistered_guests:
                    if present_barcodes[0]==new_guest_buffer[0] and name==new_guest_buffer[1] :                    
                        new_guest_buffer = (present_barcodes[0],name,new_guest_buffer[2]+1)
                        qrmatch = "Learning guest face : %s "%(args["newguestbufferth"]-new_guest_buffer[2])
                    else:
                        new_guest_buffer = (present_barcodes[0],name,0)
                        qrmatch = "Learning guest face : %s "%(args["newguestbufferth"]-new_guest_buffer[2])

                    print(new_guest_buffer)

            #[GU] Guest candidate is stable,
            # post the match between face and guest pass

            if new_guest_buffer[2] == args["newguestbufferth"]:

                # Open the g_pass to update it
                pass_filename = 'g_pass/%s/pass_%s.json'%(time.strftime("%Y%m%d"),new_guest_buffer[0])
                guest_name = ''

                print(pass_filename)
                
                with open(pass_filename, "r+") as f:

                    g_pass_json = f.read()
                    g_pass = json.loads(g_pass_json)


                    if g_pass['face_id'] == '':

                        # input "name" in the face_id
                        g_pass['face_id'] = name
                        g_pass['registered'] = time.strftime("%Y%m%d-%H%M%S")
                        guest_name = g_pass['guest']

                        print(g_pass)

                        # Save File
                        try:    
                            f.seek(0)                        
                            f.truncate()
                            f.write(json.dumps(g_pass))

                            # Enter it in memory : uuid2guestqr and uuid2guestname 
                            uuid2guestqr[name] = new_guest_buffer[0] 
                            uuid2guestname[name] = guest_name

                            if new_guest_buffer[0] in unregistered_guests:
                                del unregistered_guests[new_guest_buffer[0]]

                            print("Guest Pass MATCHED")

                        except:
                            pass
                        finally:
                            f.close()

                    else:
                     
                        print("Guest Pass MATCHING skipped")
                        print(unregistered_guests)

                



                
                #Open the u_meta file to update it (Long term memory).
                #The building will remember the name of the guest the next visit

                enc_filename = 'u_meta/%s/meta.json'%(name)

                print(enc_filename)
                
                with open(enc_filename, "r+") as f:

                    enc_meta_json = f.read()
                    enc_meta = json.loads(enc_meta_json)

                    # input user_name
                    enc_meta['user_name'] = guest_name
                    enc_meta['last_seen'] = time.strftime("%Y%m%d-%H%M%S")
                    print(enc_meta)
                    

                    # Save File
                    try:    
                        f.seek(0)                        
                        f.truncate()
                        f.write(json.dumps(enc_meta))
                        print("enc_meta UPDATED")
                    except:
                        pass
                    finally:
                        f.close()



        cv2.putText(frame, qrmatch, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        
    

    # [QR,RE] show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # [QR,RE] if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # [RE] update the FPS counter
    fps.update()

#[RE] stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# [QR,RE] close the output CSV file do a bit of cleanup
print("[INFO] cleaning up...")
csv.close()
cv2.destroyAllWindows()
vs.stop()