# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from infer_detectron2_detection import update_path
from ikomia import core, dataprocess
import copy
import os
# Your imports below
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import torch

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDetectron2DetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "COCO-Detection/faster_rcnn_R_50_C4_1x"
        self.conf_thres = 0.5
        self.cuda = True if torch.cuda.is_available() else False
        self.update = False
        self.custom_train = False
        self.cfg_path = ""
        self.weights_path = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.cuda = eval(param_map["cuda"])
        self.custom_train = eval(param_map["custom_train"])
        self.cfg_path = param_map["cfg_path"]
        self.weights_path = param_map["weights_path"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["cuda"] = str(self.cuda)
        param_map["custom_train"] = str(self.custom_train)
        param_map["cfg_path"] = self.cfg_path
        param_map["weights_path"] = self.weights_path
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2Detection(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.predictor = None
        self.cfg = None
        self.colors = None
        # Add object detection output
        self.addOutput(dataprocess.CObjectDetectionIO())
        # Create parameters class
        if param is None:
            self.setParam(InferDetectron2DetectionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get parameters :
        param = self.getParam()
        if self.predictor is None or param.update:
            if param.custom_train:
                self.cfg = get_cfg()
                self.cfg.CLASS_NAMES = None
                self.cfg.merge_from_file(param.cfg_path)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = param.weights_path
                self.class_names = self.cfg.CLASS_NAMES
                self.colors = np.array(np.random.randint(0, 255, (len(self.class_names), 3)))
                self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)

            else:
                self.cfg = get_cfg()
                config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs", param.model_name + '.yaml')
                self.cfg.merge_from_file(config_path)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name+'.yaml').replace('\\', '/'))
                self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
                self.colors = np.array(np.random.randint(0, 255, (len(self.class_names), 3)))
                self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)

            param.update = False
            print("Inference will run on "+('cuda' if param.cuda else 'cpu'))

        # Examples :
        # Get input :
        img_input = self.getInput(0)

        # Get output :
        obj_detect_out = self.getOutput(1)

        if img_input.isDataAvailable():
            obj_detect_out.init("Detectron2_Detection", 0)
            img = img_input.getImage()
            self.infer(img, obj_detect_out)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, img, obj_detect_out):
        outputs = self.predictor(img)
        if "instances" in outputs.keys():
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes
            scores = instances.scores
            classes = instances.pred_classes

            index = 0
            for box, score, cls in zip(boxes, scores, classes):
                score = float(score)
                if score >= self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    x1, y1, x2, y2 = box.numpy()
                    cls = int(cls.numpy())
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    obj_detect_out.addObject(index, self.class_names[int(cls)], score,
                                             float(x1), float(y1), w, h, self.colors[cls])
                    index += 1
        elif "proposals" in outputs.keys():
            proposals = outputs["proposals"].to("cpu")
            boxes = proposals.get_fields()["proposal_boxes"]
            objectness_logits = proposals.get_fields()["objectness_logits"]

            for i, box in enumerate(boxes):
                obj_prob = float(torch.sigmoid(objectness_logits[i]))
                x1, y1, x2, y2 = box.numpy()

                if obj_prob > self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    obj_detect_out.addObject(i, "proposal", obj_prob, float(x1), float(y1), w, h, [255, 0, 0])


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2DetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_detection"
        self.info.shortDescription = "Inference for Detectron2 detection models"
        self.info.description = "Inference for Detectron2 detection models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.iconPath = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentationLink = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, object, detection"

    def create(self, param=None):
        # Create process object
        return InferDetectron2Detection(self.info.name, param)
