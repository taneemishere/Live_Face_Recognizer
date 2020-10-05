import cv2
import numpy as np
import face_recognition

# Capture the video from the webcam
webcam = cv2.VideoCapture(0)

# Loading the known faces images.
imran_image = face_recognition.load_image_file("imrankhan.jpg")
imran_face_encoding = face_recognition.face_encodings(imran_image)[0]

donald_image = face_recognition.load_image_file("donald.jpg")
donald_face_encoding = face_recognition.face_encodings(donald_image)[0]

# Creating array of known face encodings and their names
known_face_encodings_array = [
    imran_face_encoding,
    donald_face_encoding
]
known_face_names_array = [
    "Imran Khan",
    "Donald Trump"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Reading the current frame from web-cam
    # The successful_frame_read is boolean and frame is an array
    successful_frame, frame = webcam.read()

    # if there is an error break out of the loop
    if not successful_frame:
        break

    # Resize frame of to one forth of the size for faster processing
    # (src=frame, dsize=(0, 0),   fx, fy = scale factor along the horizontal and vertical axis)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converting the frame from BGR cv2 default to RGB for face_recognition
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video

        # face_locations returns a list of tuples of found faces locations
        # in top, right, bottom, left order
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings returns a list of 128d face encodings (one for each face in the image)
        # face_encodings (face_images=rgb_small_frame,  known_face_locations= face_locations)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for index in face_encodings:
            # Check whether or not the face matches for the known faces

            # compare_faces(known_face_encodings = known_face_encodings_array,
            # face_encoding_to_check = index, tolerance=0.6)
            matches = face_recognition.compare_faces(known_face_encodings_array, index)
            name = "Unknown"

            # Using the known face with the smallest distance to the new face

            # face_distance(face_encodings = known_face_encodings_array, face_to_compare = index)
            # returns a numpy ndarray with the distance for each face in the same order as the faces array
            face_distances = face_recognition.face_distance(known_face_encodings_array, index)

            # Returns the indices of the minimum values along an axis
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names_array[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    # x, y, w, h
    # the zip() returns a zip object which is an iterator of tuples where the
    # first item in each passed iterator is paired together, and then the second
    # item in each passed iterator are paired together etc.
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Scale back up face locations since the frame we detected in was scaled to one forth size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # For putting the name over the known faces
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Display the resulting footage
    cv2.imshow('Face Recognizer', frame)

    # If q is pressed quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
webcam.release()
cv2.destroyAllWindows()
