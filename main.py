from occultus.core import Occultus
from tqdm import tqdm

# TODO: Handle not valid files

occultus = Occultus("weights/kamukha-v3.pt", show_label=True)
occultus.set_blur_type("pixel")
# For Images
occultus.detect_input()
