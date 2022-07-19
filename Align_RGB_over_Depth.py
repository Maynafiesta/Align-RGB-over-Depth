from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, \
    QRadioButton, QCheckBox, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QPixmap
import PyQt5.QtGui
from qtrangeslider import QRangeSlider

import pyrealsense2 as rs
import numpy as np
import time
import cv2
import argparse
import sys

MAX_DISTANCE = 150  # Max distance of pixel to filter with unit CM- Default : 150
MIN_DISTANCE = 30  # Min distance of pixel to filter with unit CM- Default : 30

STREAM_TYPE = 2  # 0: RGB, 1: DEPTH, 2: ALIGN - Default : 2

TEMPORAL_FILTER_FLAG = False  # Temporal filter, one of the post proccesing filters. - Default : False
SPATIAL_FILTER_FLAG = False


class Window(QWidget):
    def __init__(self, display_status: bool = False):
        super().__init__()
        self.__display_status = display_status
        grid_layout = QGridLayout()
        left_side_vertical_layout = QVBoxLayout()
        self.radiobutton_obj = self.__stream_type_radiobuttons()
        self.slider_obj = self.__create_range_slider()
        self.filters_obj = self.__filter_check_boxes()
        self.pushbuttons_obj = self.__create_control_buttons()
        left_side_vertical_layout.addWidget(self.radiobutton_obj)
        left_side_vertical_layout.addWidget(self.slider_obj)
        left_side_vertical_layout.addWidget(self.filters_obj)
        left_side_vertical_layout.addWidget(self.pushbuttons_obj)
        self.setLayout(grid_layout)
        self.setWindowTitle("SmartIR")
        # self.setFixedSize(1200, 480)

        self.disply_width = 848  # Camera frame width - Default : 640
        self.display_height = 480  # Camera frame height - Default : 480

        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        vertical_box = QVBoxLayout()
        vertical_box.addWidget(self.image_label)
        grid_layout.addLayout(left_side_vertical_layout, 0, 0)
        grid_layout.addLayout(vertical_box, 0, 1)

        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE

        self.__start_camera_stream()

    def __start_camera_stream(self):
        self.stream_thread = align_rgb_over_depth(self.__display_status)
        self.stream_thread.change_pixmap_signal.connect(self.__update_image)
        self.stream_thread.start()

    def __stop_camera_stream(self):
        self.stream_thread.stop()

    def __slider_set_visibility(self, visibility_status):
        self.slider_obj.setEnabled(visibility_status)

    def closeEvent(self, event):
        self.stream_thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def __update_image(self, cv_img):
        qt_img = self.__convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def __convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = PyQt5.QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                                  PyQt5.QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def __radio_button_rgb_selected(self, selected):
        global STREAM_TYPE
        if selected:
            STREAM_TYPE = 0
            self.__slider_set_visibility(False)

    def __radio_button_depth_selected(self, selected):
        global STREAM_TYPE
        if selected:
            STREAM_TYPE = 1
            self.__slider_set_visibility(False)

    def __radio_button_align_selected(self, selected):
        global STREAM_TYPE
        if selected:
            STREAM_TYPE = 2
            self.__slider_set_visibility(True)

    def __stream_type_radiobuttons(self):
        self.radiobutton_groupbox = QGroupBox("Select Stream Type.")
        self.horizontal_radio_buttons = QHBoxLayout()
        self.radio_button_rgb = QRadioButton(self)
        self.radio_button_depth = QRadioButton(self)
        self.radio_button_align = QRadioButton(self)

        self.radio_button_align.setChecked(True)  # Default Selected Option.

        self.radio_button_rgb.setText("RGB")
        self.radio_button_depth.setText("Depth")
        self.radio_button_align.setText("Align")

        self.horizontal_radio_buttons.addWidget(self.radio_button_rgb)
        self.horizontal_radio_buttons.addWidget(self.radio_button_depth)
        self.horizontal_radio_buttons.addWidget(self.radio_button_align)

        self.radio_button_rgb.toggled.connect(self.__radio_button_rgb_selected)
        self.radio_button_depth.toggled.connect(self.__radio_button_depth_selected)
        self.radio_button_align.toggled.connect(self.__radio_button_align_selected)
        self.radiobutton_groupbox.setLayout(self.horizontal_radio_buttons)
        return self.radiobutton_groupbox

    def __create_range_slider(self):
        self.groupBox = QGroupBox("Select the interval.")
        self.slider = QRangeSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.__change_val)
        self.slider.setFixedSize(400, 50)
        self.slider.show()

        self.label_min_text = QLabel(self)
        self.label_max_text = QLabel(self)

        self.label_min_text.setText("Min : ")
        self.label_max_text.setText("Max : ")
        self.label_min_text.setFixedSize(50, 15)
        self.label_max_text.setFixedSize(50, 15)

        self.label_value_min = QLabel(self)
        self.label_value_max = QLabel(self)
        self.label_value_min.setFixedSize(50, 15)
        self.label_value_max.setFixedSize(50, 15)

        self.horizontal_min_max_label = QHBoxLayout()
        self.horizontal_min_max_label.addWidget(self.label_min_text)
        self.horizontal_min_max_label.addWidget(self.label_value_min)
        self.horizontal_min_max_label.addWidget(self.label_max_text)
        self.horizontal_min_max_label.addWidget(self.label_value_max)

        self.vertival_box = QVBoxLayout()
        self.vertival_box.addWidget(self.slider)
        self.vertival_box.addLayout(self.horizontal_min_max_label)
        self.groupBox.setLayout(self.vertival_box)
        return self.groupBox

    def __create_control_buttons(self):
        self.create_control_buttons_groupbox = QGroupBox("Control Buttons")

        self.button_restart = QPushButton("Restart")
        self.button_restart.setToolTip("Restart Camera Connection")
        self.button_restart.clicked.connect(self.__button_restart_clicked)
        self.button_restart.setVisible(True)

        self.horizontal_buttons = QHBoxLayout()
        self.horizontal_buttons.addWidget(self.button_restart)
        self.create_control_buttons_groupbox.setLayout(self.horizontal_buttons)
        return self.create_control_buttons_groupbox

    def __button_restart_clicked(self):
        self.__stop_camera_stream()
        time.sleep(1)
        self.__start_camera_stream()

    def __filter_check_boxes(self):
        self.filter_checkbox_groupbox = QGroupBox("Post Processing Filters")
        self.checkbox_temporal_filter = QCheckBox("Temporal Filter")
        self.checkbox_temporal_filter.setChecked(False)
        self.checkbox_temporal_filter.setVisible(True)
        self.checkbox_temporal_filter.stateChanged.connect(self.__checkbox_temporal_filter_changed)

        self.checkbox_spatial_filter = QCheckBox("Spatial Filter")
        self.checkbox_spatial_filter.setChecked(False)
        self.checkbox_spatial_filter.setVisible(True)
        self.checkbox_spatial_filter.stateChanged.connect(self.__checkbox_spatial_filter_changed)

        self.horizontal_checkboxes = QHBoxLayout()
        self.horizontal_checkboxes.addWidget(self.checkbox_temporal_filter)
        self.horizontal_checkboxes.addWidget(self.checkbox_spatial_filter)
        self.filter_checkbox_groupbox.setLayout(self.horizontal_checkboxes)
        return self.filter_checkbox_groupbox

    def __checkbox_spatial_filter_changed(self):
        global SPATIAL_FILTER_FLAG
        status = self.sender()
        SPATIAL_FILTER_FLAG = status.isChecked()

    def __checkbox_temporal_filter_changed(self):
        global TEMPORAL_FILTER_FLAG
        status = self.sender()
        TEMPORAL_FILTER_FLAG = status.isChecked()

    def __change_val(self, value):
        global MAX_DISTANCE
        global MIN_DISTANCE
        self.min_distance = int((value[0] + 1) * 4.7)
        self.max_distance = int((value[1] + 1) * 4.7)
        MAX_DISTANCE = self.max_distance
        MIN_DISTANCE = self.min_distance
        self.label_value_min.setText(str(30 + self.min_distance))
        self.label_value_max.setText(str(30 + self.max_distance))


class align_rgb_over_depth(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, display_status: bool = False,
                 rgb_queue=None,
                 width: int = 640,
                 height: int = 480,
                 frame_rate: int = 30,
                 ):
        super().__init__()
        self.DEPTH_RESOLUTION_WIDTH = width
        self.DEPTH_RESOLUTION_HEIGHT = height
        self.DEPTH_FRAME_RATE = frame_rate

        self.RGB_RESOLUTION_WIDTH = width
        self.RGB_RESOLUTION_HEIGHT = height
        self.RGB_FRAME_RATE = frame_rate

        self._run_flag = True

        # self.SAMPLE_FILE_PATH = 'd435i_sample_data/d435i_walking.bag'
        self.WINDOW_TITLE = "Align RGB over Depth"
        self.DISPLAY_STATUS = display_status
        self.RGB_QUEUE = rgb_queue

        self.min_clipping_distance_in_meters = float(MIN_DISTANCE / 100)
        self.max_clipping_distance_in_meters = float(MAX_DISTANCE / 100)

        self.pipeline = rs.pipeline()
        print("Calisma araligi varsayilan ",
              "Min: ", self.min_clipping_distance_in_meters, " - ",
              "Max: ", self.max_clipping_distance_in_meters)

        self.config = rs.config()
        # rs.config.enable_device_from_file(self.config, self.SAMPLE_FILE_PATH, repeat_playback=False)
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)

        self.temp_filter = rs.temporal_filter( # Temporal Filter, to reduces temporal noise
            smooth_alpha=0.27, # The Alpha factor in an exponential moving average with Alpha=1 - no filter . Alpha = 0 - infinite filter - Default : 0.4
            smooth_delta=88.0, # Step-size boundary. - Default : 20
            persistence_control=7, # A set of predefined rules (masks). Check documantation for detail. - Default : 3 (Valid in 2/last 4)
        )

        self.spatial_filter = rs.spatial_filter( # Spatial Edge-Preserving filter, to enhance the smoothness of the reconstructed data.
            smooth_alpha=0.5, # The Alpha factor in an exponential moving average with Alpha=1 - no filter . Alpha = 0 - infinite filter - Default : 0.5
            smooth_delta=20, # Step-size boundary. - Default : 20
            magnitude=2, # Number of filter iterations. - Default : 2
            hole_fill=0, # An in-place heuristic symmetric hole-filling mode. Default : 0
        )

        self.config.enable_stream(rs.stream.color, # RGB Stream configuration.
                                  self.RGB_RESOLUTION_WIDTH,
                                  self.RGB_RESOLUTION_HEIGHT,
                                  rs.format.bgr8,
                                  self.RGB_FRAME_RATE)

        self.config.enable_stream(rs.stream.depth, # Depth Stream configuration.
                                  self.DEPTH_RESOLUTION_WIDTH,
                                  self.DEPTH_RESOLUTION_HEIGHT,
                                  rs.format.z16,
                                  self.DEPTH_FRAME_RATE)

        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.colorizer = rs.colorizer()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        if self.DISPLAY_STATUS:
            cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    def run(self):
        while self._run_flag:

            frames = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if STREAM_TYPE == 0:
                self.change_pixmap_signal.emit(color_image)
                continue
            elif STREAM_TYPE == 1:
                depth_image = aligned_depth_frame
                if TEMPORAL_FILTER_FLAG:
                    depth_image = self.temp_filter.process(depth_image)

                if SPATIAL_FILTER_FLAG:
                    depth_image = self.spatial_filter.process(depth_image)

                colorized_depth = np.asanyarray(self.colorizer.colorize(depth_image).get_data())
                self.change_pixmap_signal.emit(colorized_depth)
                continue

            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            self.min_clipping_distance_in_meters = float(MIN_DISTANCE / 100)
            self.max_clipping_distance_in_meters = float(MAX_DISTANCE / 100)
            self.max_clipping_distance = self.max_clipping_distance_in_meters / self.depth_scale
            self.min_clipping_distance = self.min_clipping_distance_in_meters / self.depth_scale

            bg_removed = np.where(
                ((depth_image_3d > self.max_clipping_distance) | (depth_image_3d < self.min_clipping_distance)) | (
                        depth_image_3d <= 0), grey_color, color_image)

            self.change_pixmap_signal.emit(bg_removed)

            if self.RGB_QUEUE:
                self.RGB_QUEUE.put(item=bg_removed, block=True)
            if self.DISPLAY_STATUS:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                rgb_over_depth = np.hstack((bg_removed, depth_colormap))
                cv2.imshow(self.WINDOW_TITLE, rgb_over_depth)
                key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()
                    break
        print("Executed...")

    def stop(self):
        self._run_flag = False
        self.pipeline.stop()
        # self.wait()


if __name__ == "__main__":
    print("*** Align RGB over Depth ***")
    parser = argparse.ArgumentParser()
    parser.add_argument("-cdmax", "--maxclippingdistance", type=float, default=1.5, required=False,
                        help="Max Clipping Distance for RGB align over Depth")
    parser.add_argument("-cdmin", "--minclippingdistance", type=float, default=0.5, required=False,
                        help="Min Clipping Distance for RGB align over Depth")
    parser.add_argument("-d", "--display", required=False, action='store_true',
                        help="Activate Stream service display.")

    args = vars(parser.parse_args())

    if args["minclippingdistance"] < 0.3 or args["maxclippingdistance"] > 5:
        print("Distance Range 30 cm to 500 cm.\n"
              "Min distance can not be smaller than 30 cm.\n"
              "Error rate in depth module increases by the distance.")
        exit(1)

    MIN_DISTANCE = int(args["minclippingdistance"] * 100) if args["minclippingdistance"] else MIN_DISTANCE
    MAX_DISTANCE = int(args["maxclippingdistance"] * 100) if args["maxclippingdistance"] else MAX_DISTANCE

    app = QApplication([])

    slider = Window(display_status=args["display"])
    slider.show()
    sys.exit(app.exec_())
