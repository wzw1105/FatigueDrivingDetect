import os
import datetime

import numpy as np
import math
import cv2
import onnxruntime
import time
import queue
import threading
import copy
import traceback
import requests

import torch
import datetime
import time

from detect import detect

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def print_with_time(log_info):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%D %H:%M:%S")
    print(time_str, log_info)

def check_in_box(centerx, centery, x1, y1, x2, y2):
    return centerx >= x1 and centerx <= x2 and centery >= y1 and centery <= y2

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

def get_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "models"))
    else:
        model_base_path = model_dir
    return model_base_path

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # 交
    w = max(0, min(x2, xx2) - max(x1, xx1))
    h = max(0, min(y2, yy2) - max(y1, yy1))
    inter_area = w * h

    # 并
    union = (x2 - x1) * (y2 - y1) + (xx2 - xx1) * (yy2 - yy1) - inter_area

    # 交并比
    return inter_area / union

def check_on_call(phone_x1, phone_y1, phone_x2, phone_y2, face_x1, face_y1, face_x2, face_y2):
    phone_center_x, phone_center_y = (phone_x1 + phone_x2) / 2, (phone_y1 + phone_y2) / 2
    face_width, face_height, face_center_x, face_center_y = face_x2 - face_x1, face_y2 - face_y1, (face_x1 + face_x2) / 2, (face_y1 + face_y2) / 2

    if abs(phone_center_x - face_center_x) / face_width < 1.0 and -0.5 < abs(phone_center_y - face_center_y) / face_height < 1.0 and (phone_y2 - phone_y1) > 0.3 * face_height:
        return True
    return False


class Tracker():
    def __init__(self, width, height, scan_every=3, silent=True, enable_debug = False, yolo_model = None, device=torch.device('cpu'), token = None, url = None):
        self.frame_count = 0
        self.width = width
        self.height = height
        self.wait_count = -1
        self.scan_every = scan_every
        self.silent = silent
        self.yolo_model = yolo_model

        # siz = (1320, 1080)
        # self.video_out = cv2.VideoWriter('./data/result_mp4s/align_result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, siz)
        self.have_prev_face = False
        self.prev_face_center_x = width
        self.prev_face_center_y = height
        self.cur_face_center_x, self.cur_face_center_y = 0, 0

        self.prev_result = [False, False, False, False]

        self.enable_debug = enable_debug
        self.VALID_POS_CHANGE_THRESHOLD = 0.4
        self.VALID_FACE_AREA_CHANGE_THRESHOLD = 0.5 #人脸面积变化阈值

        self.FACE_CHANGE_THRESHOLD = 0.2
        self.EYE_CHANGE_THRESHOLD = 0.2
        self.MOUTH_CHANGE_THRESHOLD = 0.3

        # 记录司机人脸信息
        self.driver_box_x1 = 0
        self.driver_box_x2 = 0
        self.driver_box_y1 = 0
        self.driver_box_y2 = 0
        self.driver_face_conf = 0
        self.driver_face_lmk_conf = 0
        self.phone_detect_conf = 0

        self.prev_face_box_gray = None
        self.prev_phone_detect_gray = None
        self.prev_face_x1, self.prev_face_y1, self.prev_face_x2, self.prev_face_y2 = 0, 0, 0, 0
        self.prev_phone_detect_box_x1, self.prev_phone_detect_box_x2, self.prev_phone_detect_box_y1, self.prev_phone_detect_box_y2 = 0, 0, 0, 0
        self.prev_face_width, self.prev_face_height = 0, 0

        self.prev_mouth_gray = None
        self.prev_mouth_x1, self.prev_mouth_y1, self.prev_mouth_x2, self.prev_mouth_y2 = 0, 0, 0, 0
        self.prev_left_eye_gray = None
        self.prev_left_eye_x1, self.prev_left_eye_y1, self.prev_left_eye_x2, self.prev_left_eye_y2 = 0, 0, 0, 0
        self.prev_right_eye_gray = None
        self.prev_right_eye_x1, self.prev_right_eye_y1, self.prev_right_eye_x2, self.prev_right_eye_y2 = 0, 0, 0, 0

        self.enable_gray = True

        self.device = device

        # 云端通信延迟检测
        self.delay = 2000
        self.max_delay_threshold = 1000
        self.token = token
        self.url = url
        self.is_cloud_connection_ok = True

        # 开启网络监视器线程
        self.net_state_watcher_thread = threading.Thread(target = self.net_state_watcher)
        self.net_state_watcher_thread.start()
        self.net_state_watcher_image_name = "test_net_state.png"

        # options = onnxruntime.SessionOptions()
        # options.inter_op_num_threads = 1
        # options.intra_op_num_threads = 1
        # options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # options.log_severity_level = 3
        # providersList = onnxruntime.capi._pybind_state.get_available_providers()
        # model_base_path = get_model_base_path(None)
        # self.session = onnxruntime.InferenceSession(os.path.join(model_base_path, "best.onnx"), sess_options=options, providers=providersList)

    def Init(self, width, height, scan_every, enable_gray = True):
        self.wait_count = -1
        self.prev_result = [False, False, False, False]
        self.scan_every = scan_every

        self.width = width
        self.height = height

        self.have_prev_face = False
        self.prev_face_center_x = width
        self.prev_face_center_y = height
        self.cur_face_center_x, self.cur_face_center_y = 0, 0

        self.prev_face_box_gray = None
        self.prev_mouth_gray = None
        self.prev_phone_detect_gray = None
        self.prev_left_eye_gray = None
        self.prev_right_eye_gray = None
        self.enable_gray = enable_gray

    def cloud_detect(self, image_path):
        headers = {
            "X-Auth-Token": self.token
        }
        params = {
            "input": open(image_path, 'rb')
        }
        try:
            print_with_time("Call for cloud service")
            ret = requests.post(self.url, headers=headers, files=params, timeout=1)
            self.delay = ret.elapsed.total_seconds() * 1000

            if self.delay > self.max_delay_threshold or not (200 <= ret.status_code <= 299):
                print_with_time(f"ERROR: Lose connection to cloud, delay = {self.delay}, HTTP status_code = {ret.status_code}")
                self.is_cloud_connection_ok = False
                return []
            else:
                print_with_time("Got result from cloud: " + str(ret.json()))
                print_with_time("Delay of end-to-end: " + str(self.delay))
                self.is_cloud_connection_ok = True
                return ret.json()['result']['bbox']

        except:
            print_with_time("ERROR: Fail to pose request !!!!.")
            self.is_cloud_connection_ok = False
            traceback.print_exc()
            return []

    def net_state_watcher(self):
        while True:
            if self.is_cloud_connection_ok:
                time.sleep(2)
                continue

            headers = {
                "X-Auth-Token": self.token
            }
            params = {
                "input": open(self.net_state_watcher_image_name, 'rb')
            }
            try:
                ret = requests.post(self.url, headers=headers, data=params)
                test_delay = ret.elapsed.total_seconds() * 1000

                if test_delay > self.max_delay_threshold or not (200 <= ret.status_code <= 299):
                    print_with_time(f"Watcher: Lose connection to cloud, delay = {test_delay}, HTTP status_code = {ret.status_code}")
                    self.is_cloud_connection_ok = False
                else:
                    print_with_time("Watcher: Oh, connect to cloud service successfully !!!")
                    self.is_cloud_connection_ok = True
            except:
                print_with_time("Watcher: Fail to pose request !!!!.")
                self.is_cloud_connection_ok = False

            time.sleep(5)

    # 视频一开始人脸就朝后，且后座的人脸被检测出来的case没有考虑
    def predict(self, frame, is_huawei_cloud_connected, capture = "TEST"):
        self.is_cloud_connection_ok = is_huawei_cloud_connected
        wait_time = 100

        result = None

        #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.frame = frame
        if self.enable_debug:
            self.show_frame = frame.copy()
        self.frame_count += 1

        #print("scan_every, wait_count: ", self.scan_every, self.wait_count)
        self.wait_count = (self.wait_count + 1) % self.scan_every
        if self.wait_count % self.scan_every != 0:
            #print("returnnnnnnnnnnnnnnnnnnnnn")
            return self.prev_result, result, self.is_cloud_connection_ok, self.delay

        # 判断是否发生面部变化、眼部、嘴部变化
        need_recalculate = True
        total_face_change_ratio, mouth_change_ratio, left_eye_change_ratio, right_eye_change_ratio = 0, 0, 0, 0
        if self.enable_gray:
            if self.prev_face_box_gray is not None: # 前一次检测到司机人脸（大前提）
                start = time.perf_counter()
                #print("prev box: ", self.prev_phone_detect_box_y1, self.prev_phone_detect_box_y2, self.prev_phone_detect_box_x1, self.prev_phone_detect_box_x2)
                cur_face = cv2.cvtColor(frame[self.prev_phone_detect_box_y1:self.prev_phone_detect_box_y2, self.prev_phone_detect_box_x1:self.prev_phone_detect_box_x2], cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(self.prev_face_box_gray, cur_face)
                #self.face_mse = ((cur_face - self.prev_phone_detect_gray) ** 2).mean()

                # cv2.imshow("cur_frame", frame_diff)
                face_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]
                total_area = (self.prev_phone_detect_box_y2 - self.prev_phone_detect_box_y1) * (self.prev_phone_detect_box_x2 - self.prev_phone_detect_box_x1)
                total_face_change_ratio = np.count_nonzero(face_thresh == 255) / total_area

                if self.enable_debug:
                    cv2.imshow("thresh", face_thresh)
                    cv2.waitKey(10)

                # print("white_ratio: ", 1.0 * white_num / ((self.prev_phone_detect_box_y2 - self.prev_phone_detect_box_y1) * (self.prev_phone_detect_box_x2 - self.prev_phone_detect_box_x1)), " time: ", duration_, " ms")

                if total_face_change_ratio < self.FACE_CHANGE_THRESHOLD: # 人脸变化很小
                    if self.prev_mouth_gray is not None:
                        cur_mouth = cv2.cvtColor(frame[self.prev_mouth_y1: self.prev_mouth_y2, self.prev_mouth_x1: self.prev_mouth_x2], cv2.COLOR_BGR2GRAY)
                        frame_diff = cv2.absdiff(self.prev_mouth_gray, cur_mouth)
                        mouth_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]
                        total_area = (self.prev_mouth_x2 - self.prev_mouth_x1) * (self.prev_mouth_y2 - self.prev_mouth_y1)
                        mouth_change_ratio = np.count_nonzero(mouth_thresh == 255) / total_area

                        if mouth_change_ratio < self.MOUTH_CHANGE_THRESHOLD and self.prev_left_eye_gray is not None and self.prev_right_eye_gray is not None:

                            cur_left_eye = cv2.cvtColor(frame[self.prev_left_eye_y1: self.prev_left_eye_y2, self.prev_left_eye_x1: self.prev_left_eye_x2], cv2.COLOR_BGR2GRAY)
                            frame_diff = cv2.absdiff(self.prev_left_eye_gray, cur_left_eye)
                            left_eye_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]
                            total_area = (self.prev_left_eye_x2 - self.prev_left_eye_x1) * (self.prev_left_eye_y2 - self.prev_left_eye_y1)
                            left_eye_change_ratio = np.count_nonzero(left_eye_thresh == 255) / total_area

                            cur_right_eye = cv2.cvtColor(frame[self.prev_right_eye_y1: self.prev_right_eye_y2, self.prev_right_eye_x1: self.prev_right_eye_x2], cv2.COLOR_BGR2GRAY)
                            frame_diff = cv2.absdiff(self.prev_right_eye_gray, cur_right_eye)
                            right_eye_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]
                            total_area = (self.prev_right_eye_x2 - self.prev_right_eye_x1) * (self.prev_right_eye_y2 - self.prev_right_eye_y1)
                            right_eye_change_ratio = np.count_nonzero(right_eye_thresh == 255) / total_area

                            if max(left_eye_change_ratio, right_eye_change_ratio) < self.EYE_CHANGE_THRESHOLD:
                                #print("hahahahhaha tiaoguola: %.2f %.2f %.2f %.2f" % (total_face_change_ratio, mouth_change_ratio, left_eye_change_ratio, right_eye_change_ratio))
                                need_recalculate = False
                duration_ = 1000 * (time.perf_counter() - start)

            if self.enable_debug:
                cv2.putText(self.show_frame,
                            "face_tot: %.2f mouth: %.2f left_eye: %.2f right_eye: %.2f" %
                            (total_face_change_ratio, mouth_change_ratio, left_eye_change_ratio, right_eye_change_ratio),
                            (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2)
                cv2.imshow(capture, self.show_frame)
                cv2.waitKey(wait_time)

            if need_recalculate is False:
                print_with_time("Neglect the current image!")
                return self.prev_result, result, self.is_cloud_connection_ok, self.delay

        # 人脸检测部分，每self.scan_every帧检测一次
        start = all_start = time.perf_counter()
        # result = detect(self.yolo_model, frame, 32, 640, self.enable_debug, self.device)

        if self.is_cloud_connection_ok:
            print_with_time("Using HUAWEI ModelArts for inference.")
            cv2.imwrite("call_cloud_image.png", frame)
            result = self.cloud_detect("call_cloud_image.png")
        else:
            print_with_time("Using Local device for inference.")
            #start = time.time()
            result = detect(self.yolo_model, frame, 32, 640, self.enable_debug, self.device)
            #print("inference time: ", time.time() - start)

        scrfd_duration = 1000 * (time.perf_counter() - start)
        #print("scrfd duration: ", scrfd_duration)

        # 转到距离右下角最近的人脸框
        start = time.perf_counter()
        # 找到面积最大的人脸
        max_area, driver_index = -1, -1
        for i in range(len(result)):
            x1, y1, x2, y2, conf, cls = result[i]
            if cls == 0 or cls == 1:
                cur_area = (x2 - x1) * (y2 - y1)
                if cur_area > max_area:
                    max_area = cur_area
                    driver_index = i

        for i in range(len(result)):
            x1, y1, x2, y2, conf, cls = result[i]
            if cls == 0 or cls == 1:
                if i != driver_index and conf > result[driver_index][4] and calculate_iou((x1, y1, x2, y2), (result[driver_index][0], result[driver_index][1], result[driver_index][2], result[driver_index][3])) > 0.85:
                    driver_index = i

        # 检查和上一次检测到的司机人脸坐标的变化比，以查看是不是发生了迁移
        # 注意！！！ 如果第一帧没有检测到司机人脸，而是检测到的乘客人脸，那会出问题
        if driver_index >= 0:
            self.driver_box_x1, self.driver_box_y1, self.driver_box_x2, self.driver_box_y2 = result[driver_index][:4]
            self.driver_face_conf = result[driver_index][4]
            self.cur_face_center_x, self.cur_face_center_y = (self.driver_box_x1 + self.driver_box_x2) / 2, (self.driver_box_y1 + self.driver_box_y2) / 2
            if self.have_prev_face is True:
                x_change_ratio = abs(self.prev_face_center_x - self.cur_face_center_x) / self.width
                y_change_ratio = abs(self.prev_face_center_y - self.cur_face_center_y) / self.height
                if x_change_ratio > self.VALID_POS_CHANGE_THRESHOLD or y_change_ratio > self.VALID_POS_CHANGE_THRESHOLD:
                    #print("y_change: ", abs(self.prev_face_center_y - self.cur_face_center_y), self.height)
                    if self.enable_debug:
                        cv2.rectangle(self.show_frame, (self.driver_box_x1, self.driver_box_y1), (self.driver_box_x2, self.driver_box_y2), (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(self.show_frame, "Pos_change too large: x=%.2f y=%.2f" % (x_change_ratio, y_change_ratio), (self.driver_box_x1, int(self.driver_box_y1 + 30)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print("The current detection is passenger since dis change too big， change_ratio is (x = %.2f, y = %.2f)!!" % (x_change_ratio, y_change_ratio))
                    #print("position: ", self.prev_face_center_x, self.prev_face_center_y, self.cur_face_center_x, self.cur_face_center_y)
                    driver_index = -1
                else:
                    prev_face_area = (self.prev_face_x2 - self.prev_face_x1) * (self.prev_face_y2 - self.prev_face_y1)
                    cur_face_area = (self.driver_box_x2 - self.driver_box_x1) * (self.driver_box_y2 - self.driver_box_y1)
                    face_area_change_ratio = (prev_face_area - cur_face_area ) / max(prev_face_area, cur_face_area)
                    # print("area change ratio: ", face_area_change_ratio)
                    if face_area_change_ratio > self.VALID_FACE_AREA_CHANGE_THRESHOLD:
                        if self.enable_debug:
                            cv2.rectangle(self.show_frame, (self.driver_box_x1, self.driver_box_y1),(self.driver_box_x2, self.driver_box_y2), (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(self.show_frame,"Area_change too large: %.2f" % face_area_change_ratio,(self.driver_box_x1, int(self.driver_box_y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                        # print("The currrent detection is passenger since area changes too big, change_ratio is %.2f !!", face_area_change_ratio)
                        driver_index = -1

        # 没有检测到司机人脸，定义为左顾右盼(没有检测到有没有可能是因为手机遮挡)
        if driver_index == -1:
            # print("LOW face conf!!")
            # print("no driver face detected, checking on_call...")
            on_call, yawn = False, False
            self.prev_face_box_gray = None #标记丢失脸部，但是self.have_prev_face会保持原有状态

            if self.have_prev_face is True:
                for i in range(len(result)):
                    x1, y1, x2, y2, conf, cls = result[i]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    if cls == 6 and check_on_call(x1, y1, x2, y2, self.prev_face_x1, self.prev_face_y1, self.prev_face_x2, self.prev_face_y2): #abs(center_x - self.prev_face_center_x) / self.prev_face_width < 1.0 and -0.2 < abs(self.prev_face_center_y - center_y) / self.prev_face_height < 1.0 and (y2 - y1) > 0.5 * self.prev_face_height:
                        on_call = True
                    if cls == 4 and check_in_box(center_x, center_y, self.prev_face_x1, self.prev_face_y1, self.prev_face_x2, self.prev_face_y2):
                        yawn = True
            if on_call:
                # print("ON CALL!", self.phone_detect_conf)
                state = "ON CALL"
                self.prev_result = [False, False, True, False]
            elif yawn:
                # print("yawn!!")
                state = "YAWN"
                self.prev_result = [False, True, False, False]
            else:
                self.prev_result = [False, False, False, True]
                state = "LOOK LEFT RIGHT"

            if self.enable_debug:
                (width, height), bot = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(self.show_frame, state, (int(self.prev_face_center_x - width / 2), int(self.prev_face_center_y + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow(capture, self.show_frame)
                cv2.waitKey(wait_time)
            return self.prev_result, result, self.is_cloud_connection_ok, self.delay

        # # 记录前一帧的司机人脸中心点用于下一帧司机人脸判断
        self.have_prev_face = True
        self.prev_face_width = self.driver_box_x2 - self.driver_box_x1
        self.prev_face_height = self.driver_box_y2 - self.driver_box_y1
        self.prev_face_x1, self.prev_face_y1, self.prev_face_x2, self.prev_face_y2 = self.driver_box_x1, self.driver_box_y1, self.driver_box_x2, self.driver_box_y2
        self.prev_phone_detect_box_x1, self.prev_phone_detect_box_y1 = clamp_to_im((int(self.driver_box_x1 - self.prev_face_width / 4), self.driver_box_y1), self.width, self.height)
        self.prev_phone_detect_box_x2, self.prev_phone_detect_box_y2 = clamp_to_im((int(self.driver_box_x2 + self.prev_face_width / 4), self.driver_box_y2), self.width, self.height)
        #
        # print("cur: ",self.cur_face_center_x, self.cur_face_center_y)
        self.prev_face_center_x = self.cur_face_center_x
        self.prev_face_center_y = self.cur_face_center_y
        # #print(self.driver_box_y1, self.driver_box_y2, self.driver_box_x1, self.driver_box_x2)
        if self.enable_gray:
            self.prev_face_box_gray = cv2.cvtColor(frame[self.prev_phone_detect_box_y1:self.prev_phone_detect_box_y2, self.prev_phone_detect_box_x1:self.prev_phone_detect_box_x2], cv2.COLOR_BGR2GRAY) # 前一个人脸框的灰度图
        # find_driver_box_duration = 1000 * (time.perf_counter() - start)
        #
        # # 画人脸框
        if self.enable_debug:
            cv2.rectangle(self.show_frame, (self.driver_box_x1, self.driver_box_y1), (self.driver_box_x2, self.driver_box_y2), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(self.show_frame, "FACE BBOX: %.3f" % self.driver_face_conf, (self.driver_box_x1, self.driver_box_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # 找到所有司机人脸内的状态信息

        state = [False for _ in range(7)]
        state[result[driver_index][5]] = True #正脸还是反脸
        eyes, mouth, mouth_conf = [], None, 0
        for i in range(len(result)):
            x1, y1, x2, y2, conf, cls = result[i]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if check_in_box(center_x, center_y, self.driver_box_x1, self.driver_box_y1, self.driver_box_x2, self.driver_box_y2):
                if cls > 1:
                    state[cls] = True

                if cls == 2 or cls == 3:
                    eyes.append(result[i])
                elif (cls == 4 or cls == 5) and conf > mouth_conf:
                    mouth_conf = conf
                    mouth = result[i]

            if cls == 6:
                if check_on_call(x1, y1, x2, y2, self.prev_face_x1, self.prev_face_y1, self.prev_face_x2, self.prev_face_y2):#abs(center_x - self.prev_face_center_x) / self.prev_face_width < 1.0 and -0.5 < abs(self.prev_face_center_y - center_y) / self.prev_face_height < 1.0 and (y2 - y1) > 0.3 * self.prev_face_height:
                    state[cls] = True
                #else:
                #    print("False on_call!!!!!!!!!!!!")

        close_eyes, yawn, on_call, look_left_right = state[3], state[4], state[6], state[1]
        # if state[2] is True:
        #     close_eyes = False
        # if look_left_right is True:
        #     close_eyes = False
        self.prev_result = [close_eyes, yawn, on_call, look_left_right]

        if self.enable_gray:
            left_eye, right_eye, left_pos, right_pos = None, None, 100000000, -1
            for i in range(len(eyes)):
                x1, y1, x2, y2, conf, cls = eyes[i]
                centerx, centery = (x1 + x2) / 2, (y1 + y2) / 2
                if centerx < left_pos:
                    left_eye = x1, y1, x2, y2
                    left_pos = centerx
                if centerx > right_pos:
                    right_eye = x1, y1, x2, y2
                    right_pos = centerx

            if left_eye is not None and right_eye is not None:
                self.prev_left_eye_x1, self.prev_left_eye_y1, self.prev_left_eye_x2, self.prev_left_eye_y2 = left_eye
                self.prev_left_eye_gray = cv2.cvtColor(frame[self.prev_left_eye_y1:self.prev_left_eye_y2, self.prev_left_eye_x1:self.prev_left_eye_x2], cv2.COLOR_BGR2GRAY)
                if self.enable_debug:
                    cv2.rectangle(self.show_frame, (self.prev_left_eye_x1, self.prev_left_eye_y1), (self.prev_left_eye_x2, self.prev_left_eye_y2), (255, 0, 0), 2, cv2.LINE_AA)

                self.prev_right_eye_x1, self.prev_right_eye_y1, self.prev_right_eye_x2, self.prev_right_eye_y2 = right_eye
                self.prev_right_eye_gray = cv2.cvtColor(frame[self.prev_right_eye_y1:self.prev_right_eye_y2, self.prev_right_eye_x1:self.prev_right_eye_x2], cv2.COLOR_BGR2GRAY)
                if self.enable_debug:
                    cv2.rectangle(self.show_frame, (self.prev_right_eye_x1, self.prev_right_eye_y1), (self.prev_right_eye_x2, self.prev_right_eye_y2), (0, 0, 255), 2, cv2.LINE_AA)
            if mouth is not None:
                self.prev_mouth_x1, self.prev_mouth_y1, self.prev_mouth_x2, self.prev_mouth_y2, _, _ = mouth
                self.prev_mouth_gray = cv2.cvtColor(frame[self.prev_mouth_y1:self.prev_mouth_y2, self.prev_mouth_x1:self.prev_mouth_x2], cv2.COLOR_BGR2GRAY)
                if self.enable_debug:
                    cv2.rectangle(self.show_frame, (self.prev_mouth_x1, self.prev_mouth_y1), (self.prev_mouth_x2, self.prev_mouth_y2), (255, 0, 0), 2, cv2.LINE_AA)

        if self.enable_debug:
            if any(self.prev_result) is False:
                show = 'NORMAL'
            else:
                show = ''
                if close_eyes:
                    show += 'close eyes '
                if yawn:
                    show += 'yawn '
                if on_call:
                    show += 'on_call '
                if look_left_right:
                    show += 'look_le_ri'
            (width, height), bot = cv2.getTextSize(show, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(self.show_frame, show, (int(self.cur_face_center_x - width / 2), int(self.cur_face_center_y + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow(capture, self.show_frame)
            #cv2.imshow("frame", frame)
            cv2.waitKey(wait_time)

        return self.prev_result, result, self.is_cloud_connection_ok, self.delay
