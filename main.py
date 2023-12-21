from occultus.core import Occultus
from tqdm import tqdm

occultus = Occultus("weights/kamukha-v3.pt")
occultus.set_blur_type("gaussian")
occultus.detect_video("video/news-1.mp4")
