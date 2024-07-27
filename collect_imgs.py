import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100

cap = cv2.VideoCapture(0)  # Ensure this is the correct camera index

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Define the box dimensions and position
BOX_WIDTH = 400
BOX_HEIGHT = 300
BOX_X = 100
BOX_Y = 100

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Initial prompt to start collecting data
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image")
            continue

        # Draw the box on the frame
        top_left = (BOX_X, BOX_Y)
        bottom_right = (BOX_X + BOX_WIDTH, BOX_Y + BOX_HEIGHT)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect dataset_size images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image")
            continue

        # Draw the box on the frame
        top_left = (BOX_X, BOX_Y)
        bottom_right = (BOX_X + BOX_WIDTH, BOX_Y + BOX_HEIGHT)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Crop the image to the box area
        cropped_frame = frame[BOX_Y:BOX_Y + BOX_HEIGHT, BOX_X:BOX_X + BOX_WIDTH]

        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, cropped_frame)
        print(f"Captured image {counter} for class {j}")

        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Finished collecting data.")
