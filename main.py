from occultus.core import Occultus
from tqdm import tqdm

occultus = Occultus("weights/kamukha-v3.pt", show_label=True)
occultus.set_blur_type("default")
occultus.detect_video("video/crowd.mp4")

# For videos
# occultus.detect_video()

# For Images
# occultus.detect_display()
