from occultus.core import Occultus

detect = Occultus("weights/kamukha-v2.pt")
detect.load_video("video/news-1.mp4")
detect.set_config({"conf-thres": 0.50, "track": False, "name": "inference-test"})
detect.run()
