from occultus.core import Occultus
import cv2

occultus = Occultus("weights/kamukha-v3.pt", blur_type="pixel")
# Detect opencv frame

# image = cv2.imread("video/group.jpg")

# result = occultus.detect(image)

# print(result)

# occultus.detect_image("video/group.jpg")

occultus.detect_input()

# occultus.detect_video("video/crowd.mp4")

"""for (
    frame_id,
    boxes,
) in occultus.detect_input_generator():
    print(boxes)
"""


"""for( boxes,
     frame_num,
     max_frame
) in occultus.detect_video_generator("video/mememe.mp4"):
    print(boxes, " ID:", frame_num)
"""
