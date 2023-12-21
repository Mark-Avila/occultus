from occultus.core import Occultus
from tqdm import tqdm

occultus = Occultus("weights/kamukha-v3.pt")
occultus.set_blur_type("pixel")
occultus.detect_video("video/crowd-2.mp4")
