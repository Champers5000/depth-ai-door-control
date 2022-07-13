from numpy import dtype
from scipy import spatial
import os
import cv2
from datetime import datetime

dir= os.getcwd()
print(dir)
datadir = "dataset/textfile/"
logdir = "logs/"
# Feature embedding vector of enrolled faces
enrolled_faces = []
enrolled_names = []
dt = datetime.now()
# The minimum distance between two faces
# to be called unique
authentication_threshold = 0.30

def init():
    print("init")
    for filename in os.listdir(datadir):
        with open(os.path.join(datadir, filename), 'r') as f:
            line=f.readline()
            #check for empty files
            if len(line) < 5:
                f.close()
                print(datadir+filename + " is empty, DELETING the file")
                os.remove(datadir+filename)
                continue
            #import the text file
            appendthis = []
            while line:
                if(line[0].isalpha()):
                    break
                else:
                    splitarray=line.split(",")[:-1]
                    numarray = [None]*len(splitarray)
                    for i in range(len(splitarray)):
                        numarray[i]= (float(splitarray[i]))
                    appendthis.append(numarray)
                line = f.readline()
            f.close()
            enrolled_faces.append(appendthis)
            enrolled_names.append(filename)
    print("loaded all faces from text files")

    authdir = logdir+"authlog.csv"
    if os.path.exists(authdir):
        print("Found auth logs file at "+ authdir)
    else:
        print("No auth log file found, creating at "+authdir)
        with open(os.path.join(logdir, "authlog.csv"), 'w') as f:
            f.write("Date,Time,Name,Matching File,Residual\n")
            f.close()


def enroll_face(embeddings, name):
    """
    This function adds the feature embedding
    for given face to the list of enrolled faces.
    This entire process is equivalent to
    face enrolment.
    """
    namecounter=1
    filename = ""
    while True:
        filename = dir + "/" + datadir + name+str(namecounter)+".txt"
        if os.path.exists(filename):
            namecounter += 1
        else:
            break
    print(filename)
    with open(filename, 'w') as f:
        for embedding in embeddings:
            for i in embedding:
                f.write(str(i)+",")
            f.write("\n")	
        f.write("done")
        f.close()
        for embedding in embeddings:
            # Add feature embedding to list of
            # enrolled faces
            enrolled_faces.append(embedding)
            enrolled_names.append(name)


def delist_face(embeddings):
    """
    This function removes a face from the list
    of enrolled faces.
    """
    # Get feature embedding vector for input images
    global enrolled_faces
    global enrolled_names
    deletedPerson = False
    if len(embeddings) > 0:
        for embedding in embeddings:
            # Iterate over the enrolled faces
            for i in range(len(enrolled_names)-1, 0, -1):
                # Compute distance between feature embedding
                # for input images and the current face's
                # feature embedding
                dist = spatial.distance.cosine(embedding, enrolled_faces[i])
                # If the above distance is more than or equal to
                # threshold, then add the face to remaining faces list
                # Distance between feature embeddings
                if dist <= authentication_threshold:
                    deleteAt(i)
                    deletedAface=True
        if deletedAface:
            print("Not all faces from detected person were deleted. Please manually delete them using r for better security")
    

def removenames(name):
    namelength = len(name)
    removedentries = 0
    if(name == "0"):
        for filename in os.listdir(datadir):
            os.remove(datadir+filename)
            removedentries+=1
        print("Removed " + str(removedentries) + " faces. There are no more registered faces")
        enrolled_faces.clear()
        enrolled_names.clear()
        return

    for i in range(len(enrolled_names)-1, 0, -1):
        if(enrolled_names[i][0:namelength] == name):
            deleteAt(i)
            removedentries += 1
    
    if(removedentries==0):
        print("No faces were found with name "+name)
    else:
        print("Removed " + str(removedentries) + " faces")
        

def authenticate_emb(embedding):
    """
    This function checks if a similar face
    embedding is present in the list of
    enrolled faces or not.
    """
    # Set authentication to False by default
    authentication = False
    global dt
    dt = datetime.now()
    if embedding is not None:
        # Iterate over all the enrolled faces
        for i in range(len(enrolled_faces)):
            # Compute the distance between the enrolled face's
            # embedding vector and the input image's
            # embedding vector
            dist = spatial.distance.cosine(embedding, enrolled_faces[i])
            # If above distance is less the threshold
            if dist < authentication_threshold:
                # Set the authenatication to True
                # meaning that the input face has been matched
                # to the current enrolled face
                authentication = True
                print("Authenticated: matching "+enrolled_names[i])
                
                #write authentication entry to log
                numberindex = -1
                personName = enrolled_names[i]
                for j in range(len(enrolled_names[i])):
                    if(personName[j].isalpha()):
                        continue
                    else:                            
                        numberindex=j
                        break
                if numberindex != -1:
                    personName=personName[0:numberindex]
                
                with open(os.path.join(logdir, "authlog.csv"), 'a') as f:
                    f.write(dt.strftime("%m/%d/%Y,%H:%M:%S")+','+ personName + ','+enrolled_names[i]+','+str(dist)+'\n')
                    f.close()
                break
        if authentication:
            # If the face was authenticated
            return True
        else:
            # If the face was not authenticated
            return False
    # Default
    return None

def takePicture(qJpeg, frame):
    global dt
    filename = logdir + "pics/"+dt.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(filename + "BW.jpg", frame)
    while True:
        encFrame = qJpeg.tryGet()
        if encFrame != None:
            cv2.imwrite(filename+".jpg", encFrame.getCvFrame())
        break


def deleteAt(i):
    enrolled_faces.pop(i)
    os.remove(datadir + enrolled_names[i])
    print("Removed " + enrolled_names.pop(i))