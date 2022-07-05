import Align_RGB_over_Depth
import cv2
import queue
import threading

rgb_queue = queue.Queue(maxsize=10)
align_obj = Align_RGB_over_Depth.align_rgb_over_depth(display_status=False,
                                                      min_distance_in_meter=0.80,
                                                      max_distance_in_meter=0.85,
                                                      rgb_queue=rgb_queue,
                                                      )
threading.Thread(target=align_obj.start).start()
title =  "test align"
cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
try:
    while True:
        frame = rgb_queue.get(block=True)
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    print("Sonlandirildi")
except KeyboardInterrupt:
    exit(1)

except Exception as err:
    print("hata: ", err)





