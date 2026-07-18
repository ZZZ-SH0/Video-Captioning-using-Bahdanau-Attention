

import shutil
import tqdm
import numpy as np
import cv2
import os

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

import config


def video_to_frames(video):

    path = config.temp_path  # CHANGED

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    if os.path.isabs(video):
        video_path = video
    else:
        video_path = os.path.join(
            config.video_dataset_path,
            video
        )

    print("Video path:", video_path)
    print("Exists:", os.path.exists(video_path))

    count = 0
    image_list = []

    cap = cv2.VideoCapture(video_path)

    print("Opened:", cap.isOpened())

    while cap.isOpened():

        ret, frame = cap.read()

        if ret is False:
            break

        frame_path = os.path.join(
            config.temp_path,  # CHANGED
            f"frame{count}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        image_list.append(frame_path)  # CHANGED

        count += 1

    cap.release()
   # cv2.destroyAllWindows()

    return image_list


def model_cnn_load():

    model_final = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    return model_final


def load_image(path):

    img = cv2.imread(path)

    img = cv2.resize(
        img,
        (224, 224)
    )

    return img


def extract_features(video, model):

    video_id = video.split(".")[0]

    print(video_id)
    print(f'Processing video {video}')

    try:

        image_list = video_to_frames(video)

        print("Frames extracted:", len(image_list))

        if len(image_list) == 0:
            raise ValueError(f"No frames extracted from {video}")

        samples = np.round(
            np.linspace(
                0,
                len(image_list) - 1,
                80
            )
        )

        image_list = [
            image_list[int(sample)]
            for sample in samples
        ]

        images = np.zeros(
            (
                len(image_list),
                224,
                224,
                3
            )
        )

        for i in range(len(image_list)):
            images[i] = load_image(image_list[i])

        images = preprocess_input(images)

        fc_feats = model.predict(
            images,
            batch_size=128,
            verbose=0
        )

        img_feats = np.array(fc_feats)

        print("Feature shape:", img_feats.shape)

        return img_feats

    finally:

        if os.path.exists(config.temp_path):
            shutil.rmtree(config.temp_path)
            print(
                f"[CLEANUP] Deleted temp folder: "
                f"{config.temp_path}"
            )
        else:
            print(
                "[CLEANUP] Temp folder already removed"
            )

def extract_feats_pretrained_cnn():
    """
    Extract features for all videos
    and save them as .npy files.
    """

    model = model_cnn_load()

    print("Model loaded")

    if not os.path.isdir(config.feature_path):
        os.mkdir(config.feature_path)

    video_list = os.listdir(
        config.video_dataset_path
    )

    if '.ipynb_checkpoints' in video_list:
        video_list.remove('.ipynb_checkpoints')

    saved_count = 0
    skipped_count = 0
    failed_count = 0

    total_videos = len(video_list)

    print(
        f"[START] Processing "
        f"{total_videos} videos"
    )

    for idx, video in enumerate(
        tqdm.tqdm(video_list)
    ):

        outfile = os.path.join(
            config.feature_path,
            video + '.npy'
        )

        # Skip already processed videos
        if os.path.exists(outfile):

            skipped_count += 1

            if skipped_count % 100 == 0:
                print(
                    f"[SKIP] "
                    f"{skipped_count} already exist"
                )

            continue

        try:

            img_feats = extract_features(
                video,
                model
            )

            np.save(
                outfile,
                img_feats
            )

            if os.path.exists(outfile):

                size_mb = (
                    os.path.getsize(outfile)
                    / (1024 * 1024)
                )

                saved_count += 1

                print(
                    f"[SAVED] "
                    f"{video}.npy "
                    f"({size_mb:.2f} MB)"
                )

            else:

                print(
                    f"[ERROR] Save failed: {video}"
                )

            print(
                f"[PROGRESS] "
                f"Saved={saved_count} "
                f"Skipped={skipped_count} "
                f"Failed={failed_count} "
                f"Current={idx+1}/{total_videos}"
            )

        except Exception as e:

            failed_count += 1

            print(
                f"[FAILED] {video}"
            )

            print(
                f"[ERROR] {str(e)}"
            )

    print("\n========== SUMMARY ==========")

    print(
        f"Saved   : {saved_count}"
    )

    print(
        f"Skipped : {skipped_count}"
    )

    print(
        f"Failed  : {failed_count}"
    )

    print(
        f"Total   : {total_videos}"
    )

    print("=============================")

if __name__ == "__main__":
    extract_feats_pretrained_cnn()

