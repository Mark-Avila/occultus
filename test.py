from occultus.core import Occultus
from tqdm import tqdm

detect = Occultus("weights/kamukha-v2.pt")
# detect.load_video("video/pips-2.mp4")
detect.load_stream()
detect.set_config({"conf-thres": 0.5, "flipped": False, "nolabel": True})

frames = detect.initialize()

# Assuming you know the total number of iterations beforehand``
# total_iterations = frames.numframes()

# progress_bar = tqdm(
#     total=total_iterations, desc="Blurring frames", position=0, leave=True
# )

for pred, dataset, iterables in detect.inference(frames):
    frame = detect.process(pred, dataset, iterables)

    # detect.save_video(frame, iterables)
    detect.show_frame(frame)
