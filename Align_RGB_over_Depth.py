from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QVBoxLayout, QLabel, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QPixmap
import PyQt5.QtGui
from qtrangeslider import QRangeSlider
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import sys

MAX_DISTANCE = 150
MIN_DISTANCE = 30


class Window(QWidget):
    def __init__(self):
        super().__init__()
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.create_range_slider(), 0, 0)
        self.setLayout(grid_layout)
        self.setWindowTitle("SmartIR")
        self.setFixedSize(1200, 480)

        self.disply_width = 640
        self.display_height = 480

        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.text_label = QLabel("Camera")

        vertical_box = QVBoxLayout()
        vertical_box.addWidget(self.image_label)
        vertical_box.addWidget(self.text_label)
        grid_layout.addLayout(vertical_box, 0, 1)

        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE

        self.thread = align_rgb_over_depth()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = PyQt5.QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                                  PyQt5.QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def create_range_slider(self):
        self.groupBox = QGroupBox("Align RGB over Depth")
        self.slider = QRangeSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.change_val)
        self.slider.show()

        self.label_title = QLabel(self)
        self.label_title.setText("Kullanılacak Aralığı Ayarlayınız.")

        self.label_min_text = QLabel(self)
        self.label_max_text = QLabel(self)

        self.label_min_text.setText("Min : ")
        self.label_max_text.setText("Max : ")

        self.label_value_min = QLabel(self)
        self.label_value_max = QLabel(self)

        self.horizontal_box = QGridLayout()
        self.horizontal_box.addWidget(self.label_min_text, 0, 0)
        self.horizontal_box.addWidget(self.label_value_min, 0, 1)
        self.horizontal_box.addWidget(self.label_max_text, 0, 2)
        self.horizontal_box.addWidget(self.label_value_max, 0, 3)

        self.vertival_box = QVBoxLayout()
        self.vertival_box.addWidget(self.label_title)
        self.vertival_box.addWidget(self.slider)
        self.vertival_box.addLayout(self.horizontal_box)
        self.groupBox.setLayout(self.vertival_box)

        return self.groupBox

    def change_val(self, value):
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
                 rgb_queue=None):
        super().__init__()
        self.DEPTH_RESOLUTION_WIDTH = 640
        self.DEPTH_RESOLUTION_HEIGHT = 480
        self.DEPTH_FRAME_RATE = 30

        self.RGB_RESOLUTION_WIDTH = 640
        self.RGB_RESOLUTION_HEIGHT = 480
        self.RGB_FRAME_RATE = 30

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

        self.config.enable_stream(rs.stream.color,
                                  self.RGB_RESOLUTION_WIDTH,
                                  self.RGB_RESOLUTION_HEIGHT,
                                  rs.format.bgr8,
                                  self.RGB_FRAME_RATE)

        self.config.enable_stream(rs.stream.depth,
                                  self.DEPTH_RESOLUTION_WIDTH,
                                  self.DEPTH_RESOLUTION_HEIGHT,
                                  rs.format.z16,
                                  self.DEPTH_FRAME_RATE)

        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", self.depth_scale)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        if self.DISPLAY_STATUS:
            cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    def run(self):
        while self._run_flag:
            self.min_clipping_distance_in_meters = float(MIN_DISTANCE / 100)
            self.max_clipping_distance_in_meters = float(MAX_DISTANCE / 100)
            self.max_clipping_distance = self.max_clipping_distance_in_meters / self.depth_scale
            self.min_clipping_distance = self.min_clipping_distance_in_meters / self.depth_scale

            frames = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
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

    def stop(self):
        self.pipeline.stop()
        self._run_flag = False
        self.wait()


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

    align_obj = None
    app = QApplication(sys.argv)

    slider = Window()
    slider.show()
    sys.exit(app.exec_())
