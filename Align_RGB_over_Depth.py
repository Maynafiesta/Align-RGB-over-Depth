import pyrealsense2 as rs
import numpy as np
import cv2
import argparse


class align_rgb_over_depth:
    def __init__(self, display_status: bool = False,
                 min_distance_in_meter: float = 0.5,
                 # from_file_status: bool = False,
                 max_distance_in_meter: float = 1.5,
                 rgb_queue = None):
        self.DEPTH_RESOLUTION_WIDTH = 640
        self.DEPTH_RESOLUTION_HEIGHT = 480
        self.DEPTH_FRAME_RATE = 30

        self.RGB_RESOLUTION_WIDTH = 640
        self.RGB_RESOLUTION_HEIGHT = 480
        self.RGB_FRAME_RATE = 30

        # self.SAMPLE_FILE_PATH = 'd435i_sample_data/d435i_walking.bag'
        self.WINDOW_TITLE = "Align RGB over Depth"
        self.DISPLAY_STATUS = display_status
        self.RGB_QUEUE = rgb_queue

        self.min_clipping_distance_in_meters = min_distance_in_meter
        self.max_clipping_distance_in_meters = max_distance_in_meter

        self.pipeline = rs.pipeline()
        print("Calisma araligi", "Min: ", self.min_clipping_distance_in_meters, " - Max: ",
              self.max_clipping_distance_in_meters)

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

        self.max_clipping_distance = self.max_clipping_distance_in_meters / self.depth_scale
        self.min_clipping_distance = self.min_clipping_distance_in_meters / self.depth_scale

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        if self.DISPLAY_STATUS:
            cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    def start(self):
        while True:
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

    try:
        align_obj = align_rgb_over_depth(display_status=args["display"],
                                         max_distance_in_meter=args["maxclippingdistance"],
                                         min_distance_in_meter=args["minclippingdistance"])
        align_obj.start()

    except KeyboardInterrupt:
        print("Interrupted -> Align RGB over Depth")
        align_obj.stop()
        exit(1)

