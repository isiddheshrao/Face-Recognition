import face_recognition
import cv2

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
example1_image = face_recognition.load_image_file("example1.jpeg")
example1_face_encoding = face_recognition.face_encodings(example1_image)[0]


# Load a second sample picture and learn how to recognize it.
example2_image = face_recognition.load_image_file("example2.jpg")
example2_face_encoding = face_recognition.face_encodings(example2_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    example1_face_encoding,
    example2_face_encoding
]
known_face_names = [
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    cv2.imwrite("new.png",frame)
    read_img = cv2.imread("new.png")

    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        unknown_counter = 0
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name == "Unknown":
                unknowns_name = "undefined" + str(unknown_counter) + ".png"
                (new_top, new_right, new_bottom, new_left) = (int(0.8 * top), int(1.2* right), int(1.2*bottom), int(0.8*left))
                cv2.imwrite(unknowns_name,read_img[new_top:new_bottom, new_left:new_right])
                unknown_counter += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
