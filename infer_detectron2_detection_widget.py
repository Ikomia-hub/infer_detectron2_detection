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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_detectron2_detection.infer_detectron2_detection_process import InferDetectron2DetectionParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import detectron2
import os
from detectron2 import model_zoo


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDetectron2DetectionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDetectron2DetectionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        config_paths = os.path.dirname(detectron2.__file__) + "/model_zoo"

        available_cfg = []
        for root, dirs, files in os.walk(config_paths, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                possible_cfg = os.path.join(*file_path.split(os.path.sep)[-2:])
                if "Detection" in possible_cfg and possible_cfg.endswith('.yaml') and 'rpn' not in possible_cfg:
                    try:
                        model_zoo.get_checkpoint_url(possible_cfg.replace('\\', '/'))
                    except RuntimeError:
                        continue
                    available_cfg.append(possible_cfg.replace('.yaml', ''))
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model Name")
        for model_name in available_cfg:
            self.combo_model.addItem(model_name)
        self.combo_model.setCurrentText(self.parameters.model_name)

        self.double_spin_thres = pyqtutils.append_double_spin(self.gridLayout, "Confidence threshold",
                                                              self.parameters.conf_thres, min=0., max=1., step=1e-2)
        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Cuda", self.parameters.cuda)
        self.check_custom = pyqtutils.append_check(self.gridLayout, "Load trained model with Ikomia",
                                                   self.parameters.use_custom_model)
        self.browse_cfg = pyqtutils.append_browse_file(self.gridLayout, "Browse config file (.yaml)", self.parameters.config_file)
        self.browse_weights = pyqtutils.append_browse_file(self.gridLayout, "Browse weights file (.pth)", self.parameters.model_weight_file)
        self.browse_cfg.setEnabled(self.check_custom.isChecked())
        self.browse_weights.setEnabled(self.check_custom.isChecked())
        self.check_custom.stateChanged.connect(self.on_check)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check(self, int):
        self.browse_cfg.setEnabled(self.check_custom.isChecked())
        self.browse_weights.setEnabled(self.check_custom.isChecked())

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.double_spin_thres.value()
        self.parameters.update = True
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.use_custom_model = self.check_custom.isChecked()
        self.parameters.config_file = self.browse_cfg.path
        self.parameters.model_weight_file = self.browse_weights.path
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDetectron2DetectionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_detection"

    def create(self, param):
        # Create widget object
        return InferDetectron2DetectionWidget(param, None)
