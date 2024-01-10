from occultus.core import Occultus
from tqdm import tqdm

# TODO: Handle not valid files

occultus = Occultus("weights/kamukha-v3.pt", show_label=True)
occultus.set_blur_type("default")
occultus.set_privacy_control("default")
# For Images
occultus.detect_video("video/crowd-2.mp4")
