import os
import time
import cv2
import numpy as np

import config
import model
import extract_features

# Detect whether running in Google Colab
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

SHOW_VIDEO = not IS_COLAB
SAVE_VIDEO = True


class VideoCaptionRealtime:

    def __init__(self):
        print("[INFO] Loading models...")

        self.tokenizer, self.encoder_model, self.decoder_model = (
            model.inference_model()
        )

        self.cnn = extract_features.model_cnn_load()
        self.video_folder = "/content/drive/MyDrive/Video-Captioning/Your_videos"
        self.max_length = config.max_length
        self.vocab_size = config.num_decoder_tokens

        self.index_to_word = {
            v: k for k, v in self.tokenizer.word_index.items()
        }

    def extract_features(self, filename):
        video_path = os.path.join(self.video_folder, filename)

        features = extract_features.extract_features(
            video_path,
            self.cnn
        )
        return features.reshape(1, 80, 2048)

    def greedy_search(self, features):

        encoder_outputs, h, c = self.encoder_model.predict(
            features, verbose=0
        )

        target = np.zeros((1, 1, self.vocab_size), dtype=np.float32)
        target[0, 0, self.tokenizer.word_index["bos"]] = 1

        caption = []

        for _ in range(self.max_length + 5):

            probs, h, c = self.decoder_model.predict(
                [target, encoder_outputs, h, c],
                verbose=0
            )

            probs = probs.reshape(self.vocab_size)

            idx = np.argmax(probs)

            if idx == 0:
                break

            word = self.index_to_word.get(idx)

            if word is None:
                break

            if word == "eos":
                break

            caption.append(word)

            target = np.zeros((1, 1, self.vocab_size), dtype=np.float32)
            target[0, 0, idx] = 1

        return " ".join(caption)

    def play_video(self, filename, caption):

        path = os.path.join(
            self.video_folder,
            filename
        )

        cap = cv2.VideoCapture(path)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (720, 480))

            cv2.rectangle(
                frame,
                (0, 430),
                (720, 480),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                caption,
                (10, 465),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Video Captioning",
                frame,
            )

            if cv2.waitKey(25) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_captioned_video(self, filename, caption):

        input_path = os.path.join(self.video_folder, filename)

        output_folder = os.path.join(
            os.path.dirname(__file__),
            "outputs"
        )

        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(
            output_folder,
            "captioned_" + filename
        )

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Cannot open video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, height - 70),
                (width, height),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.5,
                frame,
                0.5,
                0
            )

            cv2.putText(
                frame,
                caption,
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

        cap.release()
        writer.release()

        print("\nSaved to")
        print(output_path)

        if IS_COLAB:
            from IPython.display import Video, display
            display(Video(output_path, embed=True))

    def predict_video(self, filename):

        print(f"\nProcessing: {filename}")

        start = time.time()

        features = self.extract_features(filename)

        caption = self.greedy_search(features)

        elapsed = time.time() - start

        print("Caption :", caption)
        print(f"Inference time : {elapsed:.2f} sec")

        if SAVE_VIDEO:
            self.save_captioned_video(
                filename,
                caption
            )

        if SHOW_VIDEO:
            self.play_video(
                filename,
                caption
            )

    def run(self):

        folder = config.video_dataset_path1

        videos = sorted([
            f for f in os.listdir(folder)
            if f.endswith((".avi", ".mp4", ".mov", ".mkv"))
        ])

        if len(videos) == 0:
            print("No videos found.")
            return

        while True:

            print("\nAvailable videos:\n")

            for i, v in enumerate(videos):
                print(f"{i+1}. {v}")

            choice = input(
                "\nSelect video number (q to quit): "
            )

            if choice.lower() == "q":
                break

            try:
                idx = int(choice) - 1
                self.predict_video(videos[idx])

            except Exception as e:
                print(e)


if __name__ == "__main__":

    app = VideoCaptionRealtime()

    app.run()