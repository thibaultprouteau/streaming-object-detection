import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)


def main():
    st.header("WebRTC demo")

    object_detection_page = "Détection d'objets en temps réel."
    app_mode = object_detection_page
    st.subheader(object_detection_page)
    app_object_detection()
    
    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    #MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    #MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    #PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    #PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

#    CLASSES = [
#        "background",
#        "aeroplane",
#        "bicycle",
#        "bird",
#        "boat",
#        "bottle",
#        "bus",
#        "car",
#        "cat",
#        "chair",
#        "cow",
#        "diningtable",
#        "dog",
#        "horse",
#        "motorbike",
#        "person",
#        "pottedplant",
#        "sheep",
#        "sofa",
#        "train",
#        "tvmonitor",
#    ]
    COCO_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

    COLORS = np.random.uniform(0, 255, size=(len(COCO_CATEGORY_NAMES), 3))

#    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
#    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    box_score_thresh=DEFAULT_CONFIDENCE_THRESHOLD
)
            self._net.eval()
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, target=None, category_names=None):
            # Convert tensor to image and draw it.
            np_img = (image.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
            im = Image.fromarray(np_img)
            draw = ImageDraw.Draw(im)

            if target:
                # Make sure the required font is available
                font = ImageFont.truetype(font='Roboto-Regular.ttf', size=16)

                # Draw each bounding box in the target
            for box, label in zip(target['boxes'], target['labels']):
                box = box.detach().cpu().numpy()
                draw.rectangle(box, outline='black')
                label_str =  category_names[label.cpu().numpy()] if category_names else str(label.cpu().numpy()+2)
                draw.text((box[0], box[1]), label_str, fill=(0,128,256,256), font=font)
                result.append(Detection(name=label_str, prob=float(0.2)))
            return im, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_image()
            img_tensor = torch.as_tensor(np.array(img) / 255) # Normalize input to [0, 1]
            img_tensor = img_tensor.permute(2, 0, 1).float() # Reorder image axes to channel first
            img_tensor = img_tensor[0:3]
            detections = model([img_tensors])
            annotated_image, result = self._annotate_image(img_tensor[0], detections[0], category_names=COCO_CATEGORY_NAMES)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_image(annotated_image)

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=MobileNetSSDVideoProcessor,
        async_processing=True,
    )

    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()