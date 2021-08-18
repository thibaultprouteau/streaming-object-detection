#import asyncio
import logging
import logging.handlers
import queue
import threading
#import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import random

import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision
from torchvision.datasets.coco import CocoDetection

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
#import cv2
#import matplotlib.pyplot as plt
import numpy as np
#import pydub
import streamlit as st
#from aiortc.contrib.media import MediaPlayer

import seaborn

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

# rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:totopatate"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

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

ROOT_DIR = "/lium/raid01_b/tprouteau/streamlit/fete_science/coco/images"
ANN_FILE = "/lium/raid01_b/tprouteau/streamlit/fete_science/coco/annotations/instances_train2014.json"

# COLORS = np.random.uniform(0, 255, size=(len(COCO_CATEGORY_NAMES), 3)).astype(int)
COLORS = np.array([[int(r * 255), int(g * 255), int(b * 255)] for r,g,b in seaborn.color_palette('pastel', n_colors=len(COCO_CATEGORY_NAMES))])

FONT_SIZE = 16

def main():

    dataset_explorer_page = "Exploration du jeu de données d'apprentissage"
    object_detection_page = "Détection d'objets en temps réel."

    application = st.sidebar.selectbox(label="Application", options=[dataset_explorer_page, object_detection_page])

    if application == dataset_explorer_page:
        st.header(dataset_explorer_page)
        dataset = load_dataset(ROOT_DIR, ANN_FILE)
        app_dataset_explorer(dataset)
    elif application == object_detection_page:
        #app_mode = object_detection_page
        st.header(object_detection_page)
        app_object_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


@st.cache(allow_output_mutation=True)
def load_dataset(root_dir, annFile):
    return CocoDetection(root=root_dir, annFile=annFile)

def app_dataset_explorer(dataset):

    COCO_CATEGORY_DICT = {v:k for k,v in enumerate(COCO_CATEGORY_NAMES)}


    class DatasetExplorer():
        def __init__(self, dataset) -> None:
            self.dataset = dataset
            self.font = ImageFont.truetype(font='fonts/Roboto-Bold.ttf', size=FONT_SIZE)

        def _annotate_image(self, image, annotations):
            for annotation in annotations:
                [x,y,w,h] = annotation['bbox']
                draw = ImageDraw.Draw(image)
                category = annotation['category_id']
                color = tuple(COLORS[category])
                label_str =  COCO_CATEGORY_NAMES[category]
                text_size = self.font.getsize(label_str)
                box_color = tuple(COLORS[category])
                text_color = (0, 0, 0)
                draw.rectangle([x,y,w+x, h+y], outline=box_color)
                draw.rectangle((x, y, x + text_size[0], y + text_size[1]), fill=box_color)
                draw.text((x, y), label_str, fill=text_color, font=self.font)
            return image

        def _get_category_images_ids(self, category):
            image_ids = self.dataset.coco.getImgIds(catIds=category)
            return list(image_ids)

        def _chose_random_image(self, image_ids):
            return random.choice(image_ids)

        def get_random_image(self, category='all'):
            if category == 'all':
                category = ''
            else:
                category = COCO_CATEGORY_DICT[category]
            image_ids = self._get_category_images_ids(category)
            idx = self._chose_random_image(image_ids)
            image, annotations = self.dataset.__getitem__(self.dataset.ids.index(idx))
            return self._annotate_image(image, annotations)

    dataset_explorer = DatasetExplorer(dataset)

    with st.sidebar.form("myform"):
        category_name = st.selectbox(
            label="Catégorie d'exemple :",
            options=['all']+sorted([category_name for category_name in COCO_CATEGORY_NAMES if not '/' in category_name and not "__" in category_name])
        )
        nb_images = st.slider(label="Nombre d'images", min_value=1, max_value=10, value=5)
        button = st.form_submit_button("Afficher")
        if button:
            pass
    st.subheader(f"Images de la catégorie : {category_name}")
    images = [dataset_explorer.get_random_image(category=category_name) for i in range(nb_images)]
    st.image([np.array(image) for image in images])

@st.cache(allow_output_mutation=True)
def load_model(device, confidence_threshold):
    _net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        box_score_thresh=0.1 #0.001
    )
    _net.to(device)
    _net.eval()
    # print(_net.__dir__)
    # print(dir(_net))
    # _net.box_score_thresh = 0.1
    return _net


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.8

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"
        invert_image: bool

        def __init__(self) -> None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self._net = load_model(self.device, self.confidence_threshold)
            self.result_queue = queue.Queue()
            self.invert_image = False

            self.old_counter = None
            self.font = ImageFont.truetype(font='fonts/Roboto-Bold.ttf', size=FONT_SIZE)

        def _annotate_image(self, image, target=None, category_names=None):
            # Convert tensor to image and draw it.
            result: List[Detection] = []
            np_img = (image.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
            im = Image.fromarray(np_img)
            draw = ImageDraw.Draw(im)

            # Draw each bounding box in the target
            for box, label, score in zip(target['boxes'], target['labels'], target['scores']):
                if score < self.confidence_threshold:
                    continue
                box = box.detach().cpu().numpy()
                category = label.cpu().numpy()
                category_name =  COCO_CATEGORY_NAMES[category] if COCO_CATEGORY_NAMES else str(category)
                label_str = f"{category_name} {score:1.2f}"
                text_size = self.font.getsize(label_str)
                box_color = tuple(COLORS[category])
                text_color = (0, 0, 0)
                draw.rectangle(box, outline=box_color)
                draw.rectangle((box[0], box[1], box[0] + text_size[0], box[1] + text_size[1]), fill=box_color)
                draw.text((box[0], box[1]), label_str, fill=text_color, font=self.font)
                result.append(Detection(name=category_name, prob=round(float(score), 2)))
            return im, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_image()
            if self.invert_image:
                image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_tensor = torch.as_tensor(np.array(image) / 255) # Normalize input to [0, 1]
            img_tensor = img_tensor.to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1).float() # Reorder image axes to channel first
            img_tensor = img_tensor[0:3]
            if self.old_counter == None or self.old_counter >= 8:
                self.old_counter = 0
                self.old_detection = self._net([img_tensor])
            self.old_counter += 1
            # detections = self._net([img_tensor])
            annotated_image, result = self._annotate_image(img_tensor, self.old_detection[0], category_names=COCO_CATEGORY_NAMES)

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
        webrtc_ctx.video_processor.confidence_threshold = st.slider(label="Seuil de confiance minimal", step=0.05, min_value=0.1, max_value=1.0, value=DEFAULT_CONFIDENCE_THRESHOLD)
        webrtc_ctx.video_processor.invert_image = st.sidebar.checkbox("Inverser l'image", value=True)

    if st.sidebar.checkbox("Montrer les objets détectés", value=True):
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

    st.sidebar.markdown(
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
    fsevents_logger.setLevel(logging.INFO)

    main()
