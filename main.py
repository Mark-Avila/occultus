from occultus.core import Occultus
from tqdm import tqdm

# TODO: Handle not valid files

occultus = Occultus("weights/kamukha-v3.pt", show_label=True)
occultus.set_blur_type("gaussian")
occultus.detect_video(
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"
)
# For videos
# occultus.detect_video()

# For Images
# occultus.detect_display()
