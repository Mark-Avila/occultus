from occultus.core import Occultus
from tqdm import tqdm

detect = Occultus("weights/kamukha-v3.pt")
detect.load_video("video/news-1.mp4")
# detect.load_stream()
detect.set_config(
    {
        "conf-thres": 0.50,
        "flipped": False,
        "nolabel": False,
    }
)


detect.set_privacy_control("all")
detect.set_blur_type("pixel")

# detect.run()

frames = detect.initialize()

total_iterations = frames.numframes()

progress_bar = tqdm(
    total=total_iterations, desc="Blurring frames", position=0, leave=True
)

for pred, dataset, iterables in detect.inference(frames):
    [frame, dets] = detect.process(pred, dataset, iterables)

    # detect.show_frame(frame)
    detect.save_video(frame, iterables)
    progress_bar.update()

progress_bar.close()
