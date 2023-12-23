from occultus.core import Occultus
from tqdm import tqdm

occultus = Occultus("weights/kamukha-v3.pt", output_create_folder=False)
occultus.set_blur_type("pixel")
occultus.detect_video("video/mememe.mp4")

# For videos
# occultus.detect_video()

# For Images
# occultus.detect_display()
