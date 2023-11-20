from occultus.core_temp import Occultus

detect = Occultus("weights/kamukha-v2.pt")
detect.load_stream()
detect.set_config(
    {
        "conf-thres": 0.5,
    }
)

# frames = detect.initialize_model()

# for pred, dataset, iterables in detect.run_inference(frames):
#     detect.process_preds(pred, dataset, iterables)

detect.run()

# detect.run_inference(frames)

# detect.append_id(1)
# detect.run()
