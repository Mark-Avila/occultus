from occultus.core_temp import Occultus
from tqdm import tqdm

detect = Occultus("weights/kamukha-v2.pt")
detect.load_video("video/pips-2.mp4")
# detect.load_stream()
detect.set_config({"conf-thres": 0.5, "flipped": False, "nolabel": True})

frames = detect.initialize_model()

# Assuming you know the total number of iterations beforehand
total_iterations = frames.numframes()

progress_bar = tqdm(
    total=total_iterations, desc="Processing Frames", position=0, leave=True
)

for pred, dataset, iterables in detect.run_inference(frames):
    frame = detect.process_preds(pred, dataset, iterables)
    detect.save_video(frame, iterables)

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()
