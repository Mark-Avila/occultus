from occultus.core import Occultus
from tqdm import tqdm

occultus = Occultus("weights/kamukha-v3.pt")
occultus.set_blur_type("pixel")

occultus.detect_input()

# For videos
# occultus.detect_video()

# For Images
# occultus.detect_display()
