from scipy import spatial
import os

datadir = "dataset/textfile/"
# Feature embedding vector of enrolled faces
enrolled_faces = []
enrolled_names = []

namecounter=0
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

            '''
            numberindex = -1
            for i in range(len(filename)):
                if(filename[i].isalpha()):
                    continue
                else:
                    numberindex=i
                    break
            '''
            
            


    print("loaded all faces from text files")

def enroll_face(embeddings, name):
    """
    This function adds the feature embedding
    for given face to the list of enrolled faces.
    This entire process is equivalent to
    face enrolment.
    """
    global namecounter
    namecounter = namecounter+1
    filename = name+str(namecounter)+".txt"
    with open(datadir+filename, 'w') as f:
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
    if len(embeddings) > 0:
        for embedding in embeddings:
            # List of faces remaining after delisting
            remaining_faces = []
            # Iterate over the enrolled faces
            for idx, face_emb in enumerate(enrolled_faces):
                # Compute distance between feature embedding
                # for input images and the current face's
                # feature embedding
                dist = spatial.distance.cosine(embedding, face_emb)
                # If the above distance is more than or equal to
                # threshold, then add the face to remaining faces list
                # Distance between feature embeddings
                # is equivalent to the difference between
                # two faces
                if dist >= authentication_threshold:
                    remaining_faces.append(face_emb)
            # Update the list of enrolled faces
            enrolled_faces = remaining_faces


def authenticate_emb(embedding):
    """
    This function checks if a similar face
    embedding is present in the list of
    enrolled faces or not.
    """
    # Set authentication to False by default
    authentication = False

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
        '''
        for face_emb in enrolled_faces:
            # Compute the distance between the enrolled face's
            # embedding vector and the input image's
            # embedding vector
            dist = spatial.distance.cosine(embedding, face_emb)
            # If above distance is less the threshold
            if dist < authentication_threshold:
                # Set the authenatication to True
                # meaning that the input face has been matched
                # to the current enrolled face
                authentication = True
        '''
        if authentication:
            # If the face was authenticated
            return True
        else:
            # If the face was not authenticated
            return False
    # Default
    return None
