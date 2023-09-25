import json
import os
import datetime
import traceback

import numpy as np
import math
import cv2
import onnxruntime
import time
import queue
import threading
import copy
import requests

import torch

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
    def __init__(self, width, height, scan_every=3, silent=True, enable_debug = False, yolo_model = None):
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

        # 开启云端模型调用时延监控线程
        self.delay_detector = DelayDetectThread()

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

    # 视频一开始人脸就朝后，且后座的人脸被检测出来的case没有考虑
    def predict(self, frame, capture = "IMG", device=torch.device('cpu')):
        #cv2.imshow("test", frame)
        #cv2.waitKey(10)
        #print("type frame: ", type(frame), frame.shape)
        #print("frame shape", frame.shape)
        wait_time = 100
        result = None

        #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.frame = frame
        if self.enable_debug:
            self.show_frame = frame.copy()
        self.frame_count += 1
        scrfd_duration = 0.0
        find_driver_box_duration = 0.0
        cut_box_for_landmark_duration = 0.0
        cut_box_for_phone_detect_duration = 0.0
        landmark_duration = 0.0
        phone_detect_duration = 0.0
        post_process_duration = 0.0

        #print("scan_every, wait_count: ", self.scan_every, self.wait_count)
        self.wait_count = (self.wait_count + 1) % self.scan_every
        if self.wait_count % self.scan_every != 0:
            return self.prev_result, result

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
                return self.prev_result, result

        # 人脸检测部分，每self.scan_every帧检测一次
        start = all_start = time.perf_counter()
        if delay < threshold:
            result = cloud_detect(frame)
            if result is None:
                result = detect(self.yolo_model, frame, 32, 640, self.enable_debug, device)
        else:
            result = detect(self.yolo_model, frame, 32, 640, self.enable_debug, device)
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
            return self.prev_result, result

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
        if state[2] is True:
            close_eyes = False
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

        return self.prev_result, result


def cloud_detect(frame):
    headers = {"X-Auth-Token": token}
    params = {
        "input": frame.tolist()
    }
    global delay
    try:
        ret = requests.post(url, headers=headers, data=params)
        print("answer for cloud model: " + str(ret.json()))
        delay = ret.elapsed.microseconds / 1000
        print("delay from requesting cloud model: " + str(delay))
        if delay > threshold:
            print("using local model")
        return ret.json()['result']['bbox']
    except:
        print("connection failed")
        traceback.print_exc()
        print("using local model")
        delay = 9999
        return None


# 云端通信延迟检测
delay = 2000
threshold = 1000
token = "MIIRbAYJKoZIhvcNAQcCoIIRXTCCEVkCAQExDTALBglghkgBZQMEAgEwgg9+BgkqhkiG9w0BBwGggg9vBIIPa3sidG9rZW4iOnsiZXhwaXJlc19hdCI6IjIwMjMtMDktMDVUMDU6NDI6NDIuODQxMDAwWiIsIm1ldGhvZHMiOlsicGFzc3dvcmQiXSwiY2F0YWxvZyI6W10sInJvbGVzIjpbeyJuYW1lIjoidGVfYWRtaW4iLCJpZCI6IjAifSx7Im5hbWUiOiJ0ZV9hZ2VuY3kiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3JlcF9hY2NlbGVyYXRpb24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfZGlza0FjYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Rzc19tb250aCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29ic19kZWVwX2FyY2hpdmUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfc3BvdF9pbnN0YW5jZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Jkc19tYXJpYWRiX29idCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Jjc19uZXNfc2ciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLXNvdXRoLTRjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZGVjX21vbnRoX3VzZXIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9pbnRsX29hIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY2JyX3NlbGxvdXQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9mbG93X2NhIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX29sZF9yZW91cmNlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfd2VsaW5rYnJpZGdlX2VuZHBvaW50X2J1eSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Nicl9maWxlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaWRtZV9saW5reF9mb3VuZGF0aW9uIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZG1zLXJvY2tldG1xNS1iYXNpYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Rtcy1rYWZrYTMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9zbnQ5YmlubCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2VkZ2VzZWNfb2J0IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb2JzX2R1YWxzdGFjayIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29ic19kZWNfbW9udGgiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3Jlc3RvcmUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfYzZhIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfRUNfT0JUIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfa29vcGhvbmUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9tdWx0aV9iaW5kIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc21uX2NhbGxub3RpZnkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vcmdpZF9jYSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfYXAtc291dGhlYXN0LTNkIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaWFtX2lkZW50aXR5Y2VudGVyX2ludGwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3Byb2dyZXNzYmFyIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY2VzX3Jlc291cmNlZ3JvdXBfdGFnIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX29mZmxpbmVfYWM3IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZXZzX3JldHlwZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2tvb21hcCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2V2c19lc3NkMiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2V2c19wb29sX2NhIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfcGVkYV9zY2hfY2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLXNvdXRod2VzdC0yYiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2h3Y3BoIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX29mZmxpbmVfZGlza180IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaHdkZXYiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vcF9nYXRlZF9jYmhfdm9sdW1lIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc21uX3dlbGlua3JlZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2RhdGFhcnRzaW5zaWdodCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2h2X3ZlbmRvciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Vjc19vbl9hd3NfaW50bCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfY24tbm9ydGgtNGUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF93YWZfY21jIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9jbi1ub3J0aC00ZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19hYzciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lcHMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3Jlc3RvcmVfYWxsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3JnYW5pemF0aW9uc19pbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfcmRzX21hcmlhZGJfb2J0X0ludGwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lZHMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9pYW1faWRlbnRpdHljZW50ZXJfY24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLW5vcnRoLTRmIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9zZnNfbGlmZWN5Y2xlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3BfZ2F0ZWRfcm91bmR0YWJsZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfYXAtc291dGhlYXN0LTFlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9ydS1tb3Njb3ctMWIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2FwLXNvdXRoZWFzdC0xZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfYXAtc291dGhlYXN0LTFmIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc21uX2FwcGxpY2F0aW9uIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3NlX2dhdGV3YXkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9zbnQ5Ymk2bCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX0lQRENlbnRlcl9DQV8yMDIzMDgzMCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3JhbSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29yZ2FuaXphdGlvbnMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfZ3B1X2c1ciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29wX2dhdGVkX21lc3NhZ2VvdmVyNWciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9yaV9kd3MiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfcmkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9tZ2MiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9zbnQ5YiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfcnUtbm9ydGh3ZXN0LTJjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfcmFtX2ludGwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9pZWZfcGxhdGludW0iLCJpZCI6IjAifV0sInByb2plY3QiOnsiZG9tYWluIjp7Im5hbWUiOiJodzQ2MzAyOTIzIiwiaWQiOiIwODJmMzRiNTAxODAyNThlMGZiYWMwMTM4MDg1NzE4MCJ9LCJuYW1lIjoiY24tbm9ydGgtMSIsImlkIjoiMDgyZjM0YjY2MzAwMjU5NTJmMjNjMDEzOTVhYWQ3NGEifSwiaXNzdWVkX2F0IjoiMjAyMy0wOS0wNFQwNTo0Mjo0Mi44NDEwMDBaIiwidXNlciI6eyJkb21haW4iOnsibmFtZSI6Imh3NDYzMDI5MjMiLCJpZCI6IjA4MmYzNGI1MDE4MDI1OGUwZmJhYzAxMzgwODU3MTgwIn0sIm5hbWUiOiJodzQ2MzAyOTIzIiwicGFzc3dvcmRfZXhwaXJlc19hdCI6IiIsImlkIjoiMDgyZjM0YjVhMjAwMjYxYjFmYWNjMDEzN2NlMjFiZmMifX19MYIBwTCCAb0CAQEwgZcwgYkxCzAJBgNVBAYTAkNOMRIwEAYDVQQIDAlHdWFuZ0RvbmcxETAPBgNVBAcMCFNoZW5aaGVuMS4wLAYDVQQKDCVIdWF3ZWkgU29mdHdhcmUgVGVjaG5vbG9naWVzIENvLiwgTHRkMQ4wDAYDVQQLDAVDbG91ZDETMBEGA1UEAwwKY2EuaWFtLnBraQIJANyzK10QYWoQMAsGCWCGSAFlAwQCATANBgkqhkiG9w0BAQEFAASCAQAOOaCP5wG+BPMaAWCDraREwCaYdleXYzP0TsbaLrSMPGCwjGRLBDpJQYhnm-xlc1fxlGD300V85py+w7CJgOdJZYFudwGhF8sJ9Rd8M7+KwIK6Lt995noc+BgNGlBgFYuBv3kWInh1rFQadwn-hw+E999iu3vqr9XNGEtWu1yvep38volS-XWKn5-5hSq588gwNEaMsij4QYQcIu5dDmpGQlcGVOc3uAbjQDIVHCM24M4bA3opKgyEmyhd6AMRGZo9g8SWphkHjtELDoRqqBQ8CJIhnsuaPXXibdF+3Uhkynri8tm6EMozgiX-5sO9aoGOynqR3KmdvNjEXrMrkc87"
# CPU
# url = "https://2223e6fbd72a45b08c01865f13edec83.apig.cn-north-4.huaweicloudapis.com/v1/infers/6dcaae77-d18a-4654-8056-ada9bbaeca38"
# GPU
url = "https://08ce9aa27a8e4a368cacc66e8814369c.apig.cn-north-4.huaweicloudapis.com/v1/infers/4366ed17-bebe-421d-a840-508aca07abc7"
class DelayDetectThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.start()

    def start(self):
        super().start()

    def run(self):
        global delay
        while True:
            if delay <= threshold:
                continue
            headers = {"X-Auth-Token": token}
            params = {
                "input": json.dumps(cv2.imread("D:/project_files/Python/Huawei/CloudOnePic/test.png").tolist())
            }
            try:
                ret = requests.post(url, headers=headers, data=params)
                delay = ret.elapsed.microseconds/1000
                print("delay from requesting cloud model: " + str(delay))
                if delay <= threshold:
                    print("using cloud model")
            except:
                print("connection failed")
                traceback.print_exc()
                delay = 9999
            time.sleep(5)  # 1min检测一次与云端的网络延迟
