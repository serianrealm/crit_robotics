import os

import torch
import numpy as np
import openvino as ov

from ..models import Yolov10PoseModel
from ...solutions.nms import non_max_suppresson


class Yolov10PosePipeline:
    def __init__(self,
                 model_path,
                 image_size):
        model = Yolov10PoseModel.from_pretrained(
            model_path,
            use_safetensors=False,
            weights_only=True,
            dtype=torch.float32
        ).export()

        dummy_inputs = torch.randn((1, 3, *image_size)).to(model.dtype)

        with torch.inference_mode(True):
            model(dummy_inputs) # dry run

            intermediate_model = ov.convert_model(
                model, 
                input=[dummy_inputs.shape],
                example_input=dummy_inputs
            )

        core = ov.Core()
        core.set_property({
            "CACHE_DIR": os.path.expanduser("~/.cache/openvino"),
            "PERFORMANCE_HINT": "LATENCY",
        })

        ppp = ov.preprocess.PrePostProcessor(intermediate_model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")).set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(
            ov.Type.f32).resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR).scale(255.0)
        
        intermediate_model = ppp.build()

        self.model = ov.compile_model(intermediate_model, device_name="AUTO")


    def __call__(self, images:np.ndarray|list[np.ndarray], **kwargs) -> list[np.ndarray]:
        if not isinstance(images, list):
            images = [images[None, :]]

        pixel_values = np.concat(images)

        prediction = self.model([pixel_values])[self.model.output(0)]

        output = non_max_suppresson(prediction, conf_thres=0.7, iou_thres=0.3)

        output = [out.cpu().numpy() for out in output]
        return output