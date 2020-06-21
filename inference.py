import os
import cv2
import pickle
import numpy as np

from flask import Flask, json
import threading

from utils.train import train_svm_classifier
from utils.data import encode_labels, flatten_dataset, augment_data, process_images, get_ideal_pose
from utils.helpers import get_pose, convert, draw_pose
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from joblib import dump, load

poses = [{"confidence": 0, "name": ""}]

api = Flask(__name__)

@api.route('/poses', methods=['GET'])
def get_poses():
  return json.dumps(poses)

def live_inference(rate=5):
    """
    Run live inference using the webcam. Plot polar coordinates of
    the estimated pose and print prediction to terminal.
    """
    global poses

    # if there is not a classifier model, train the model then use the saved file
    # if not os.path.exists('assets/classifier.pkl'):
    if not os.path.exists('assets/classifier.joblib') or not os.path.exists('assets/ideal_poses.joblib') or not os.path.exists('assets/classes.joblib'):
        f, l, _ = process_images()
        f, l = augment_data(f, l, data_size=500)  # noqa (ambiguous variable name)
        f = flatten_dataset(f)
        l, name_map = encode_labels(l)

        ideal_poses = get_ideal_pose(f, l, name_map)

        dump(ideal_poses, 'assets/ideal_poses.joblib')
        dump(name_map, 'assets/classes.joblib')

        _, _ = train_svm_classifier(f, l, 'assets/classifier.joblib')

    #with open('assets/classifier.pkl', 'rb') as fil:
    #    classifier = pickle.load(fil)
    classifier = load('assets/classifier.joblib')
    ideal_poses = load('assets/ideal_poses.joblib')
    name_map = load('assets/classes.joblib')

    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111, projection='polar')

    # Run the pose detection using the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 656)
    cap.set(4, 368)

    if cap.isOpened():
        is_capturing, frame = cap.read()
    else:
        is_capturing = False

    current_prediction = 'shavasana'
    current_frame = 0
    while is_capturing:
        try:
            is_capturing, frame = cap.read()
            cv2.imwrite('assets/image.jpg', frame)

            if current_frame % rate == 0 or current_frame == 0:
                preds, img = get_pose()

                # make sure that we have enough vectors to accurately classify something
                if preds['predictions'] != [] and len(preds['predictions'][0]['pose_lines']) > 10:
                    coordinates = np.array([[d['x'], d['y']] for d in preds['predictions'][0]['body_parts']], dtype=np.float32)
                    frame = convert(coordinates)
                    missing_vals = 19 - frame.shape[0]
                    frame = np.concatenate([frame, np.zeros([missing_vals, 2])])
                    nx, ny = frame.shape
                    reshaped_frame = frame.reshape((1, nx*ny))

                    # make predictions
                    prediction = classifier.predict(reshaped_frame)
                    confidence = classifier.predict_proba(reshaped_frame)
                    current_prediction = name_map[prediction[0]]
                    print("{}%  confident that the pose is {}".format(confidence[0][prediction[0]], current_prediction))
                    poses = [{"confidence": confidence[0][prediction[0]], "name": current_prediction}]

                    # check cosine similarity to the ideal pose
                    # ideal = np.array(ideal_poses[current_prediction])
                    # closeness = cosine_similarity(ideal, reshaped_frame)
                    # print('SIMILARITY TO IDEAL {}\n{}'.format(current_prediction,closeness))

                    # draw frame with new predictions
                    overlaid_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    draw_pose(preds, overlaid_img)
                    cv2.imshow('Frame', overlaid_img)

                    # plot the polar chart
                    ax.cla()
                    ax.set_theta_zero_location("W")
                    colors = cm.rainbow(np.linspace(0, 1, len(frame[:, 0])))
                    ax.scatter(frame[:, 1], frame[:, 0], cmap='hsv', alpha=0.75, c=colors)
                    fig.canvas.draw()

                else:
                    cv2.imshow('Frame', frame)

            else:
                draw_pose(preds, frame)
                cv2.imshow('Frame', frame)

            current_frame += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    t1 = threading.Thread(target=live_inference)
    t2 = threading.Thread(target=get_poses)
    t3 = threading.Thread(target=api.run)

    # starting thread 1 
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()