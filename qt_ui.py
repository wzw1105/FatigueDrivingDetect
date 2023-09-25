import math
import sys
import threading
import time
import os
import requests
import re

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QCloseEvent, QFont, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton

from models.common import DetectMultiBackend
from tracker import Tracker, print_with_time
from utils1.plots import Annotator, colors

# 语音多线程类
class soundThread(QThread):
    def __init__(self, sound_player):
        super().__init__()
        self.sound_player = sound_player
        self.running = True
    
    def is_running(self):
        return self.running

    def stop(self):
        self.running = False

    def run(self):
        print("start play")
        self.sound_player.play()
        time.sleep(5)
        self.sound_player.stop()
        self.running = False

class heartbeatThread(QThread):
    stateSignal = pyqtSignal(bool)
    def __init__(self, verify_code, APIgateway):
        super().__init__()
        self.running = True
        self.verificationCode = verify_code
        self.APIgateway = APIgateway

    def run(self):
        while self.running:
            try:
                ret = requests.put("{}/device/heartbeat?verificationCode={}".format(self.APIgateway, self.verificationCode))
                print('heartbeat')
                print("answer for heartbeat: " + str(ret.json()))
                is_manage_connected = (200 <= ret.status_code <= 299)
                self.stateSignal.emit(is_manage_connected)
            
            except:
                is_manage_connected = False
                self.stateSignal.emit(is_manage_connected)
            time.sleep(5)  # 心跳周期1min

    def stop(self):
        self.running = False

class alertThread(QThread):
    stateSignal = pyqtSignal(bool)
    def __init__(self, verify_code, APIgateway, type: int):
        super().__init__()
        self.type = type
        self.userInfo = "王志伟 13637959181"
        self.verificationCode = verify_code
        self.APIgateway = APIgateway
    def getOutterIP(self):
        ip = ''
        try:
            print("getting ip...")
            res = requests.get('https://myip.ipip.net', timeout=2).text
            ip = re.findall(r'(\d+\.\d+\.\d+\.\d+)', res)
            ip = ip[0] if ip else ''
        except:
            print_with_time("Error in getting IP")
        return ip
    def getLocation(self, ip_address):
        ret = requests.get(f"https://restapi.amap.com/v3/ip?ip={ip_address}&output=json&key=7adc97f7624ef1865767693e1b183478", timeout=2)
        res = ret.json()
        if ret.status_code == 200:
            return res['province'] + ' ' + res['city']
        else:
            return "未知"
        
    def run(self):
        try:
            print("start alert")
            params = {
                "deviceVerificationCode": self.verificationCode,
                "type": self.type,
                "userInfo": self.userInfo,
                "ip": self.getLocation(self.getOutterIP()),
                # TODO 添加用户名、联系方式
                "time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime(time.time()))
            }
            print("alert type-" + str(self.type))
            ret = requests.post("{}/alert/".format(self.APIgateway), data=params)
            is_manage_connected = (200 <= ret.status_code <= 299)
            self.stateSignal.emit(is_manage_connected)
            print("answer for alert submission: " + str(ret.json()))
        except Exception as e:
            print(e)
            print("Error in posting alert info")
            is_manage_connected = False
            self.stateSignal.emit(is_manage_connected)
            self.stateSignal.emit(is_manage_connected)

class huaweiCloudWatcherThread(QThread):
    stateSignal = pyqtSignal(bool)
    def __init__(self, token, url):
        self.token = token
        self.url = url
        self.net_state_watcher_image_name = "test_net_state.png"
        self.is_cloud_connection_ok = False
        self.max_delay_threshold = 2000
        super().__init__()
    def run(self):
        while True:
            #clear
            # print("is_cloud_connection_ok: ", self.is_cloud_connection_ok)
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
                ret = requests.post(self.url, headers=headers, files=params)
                test_delay = ret.elapsed.total_seconds() * 1000

                if test_delay > self.max_delay_threshold or not (200 <= ret.status_code <= 299):
                    print_with_time(f"Watcher: Lose connection to cloud, delay = {test_delay}, HTTP status_code = {ret.status_code}")
                    self.is_cloud_connection_ok = False
                else:
                    print_with_time("Watcher: Oh, connect to cloud service successfully !!!")
                    self.is_cloud_connection_ok = True
            except Exception as e:
                print(e)
                print_with_time("Watcher: Fail to pose request !!!!.")
                self.is_cloud_connection_ok = False
            self.stateSignal.emit(self.is_cloud_connection_ok)
            time.sleep(5)
    def setCloudState(self, new_state):
        #print("set is_cloud_connection_ok: ", new_state)
        self.is_cloud_connection_ok = new_state

class updateImageThread(QThread):
    imgSignal = pyqtSignal(QPixmap)
    stateSignal = pyqtSignal(bool)
    delaySignal = pyqtSignal(int)
    def __init__(self, token, url, verify_code, APIgateway):
        super().__init__()
        self.running = True

        self.sound_thread = None
        self.bbox = []
        self.loop_array_length = 20000
        self.presum = [[0] * 5 for j in range(self.loop_array_length)]
        self.previous_frames_result = [[0] * 5 for j in range(self.loop_array_length)]
        self.previous_frames_result_time = [0.0 for j in range(self.loop_array_length)]
        self.store_index = 0

        self.token = token
        self.url = url
        self.verify_code = verify_code
        self.APIgateway = APIgateway

        # 华为云连接状态监视
        self.is_huawei_cloud_connected = True
        self.huawei_cloud_watcher_thread = huaweiCloudWatcherThread(self.token, self.url)
        self.huawei_cloud_watcher_thread.stateSignal.connect(self.update_huawei_cloud_connect_state)
        self.huawei_cloud_watcher_thread.start()
        self.detect_on = False
    
    # 监视器观察网络状态
    def update_huawei_cloud_connect_state(self, is_huawei_cloud_connected):
        self.is_huawei_cloud_connected = is_huawei_cloud_connected

    def load_mp3(self):
        url = ['closed_eyes.mp3', 'yawn.mp3', 'on_call.mp3', 'look_left_right.mp3']
        self.qMediaContents = [QMediaContent(QUrl.fromLocalFile('D:\\PycharmProjects\\EndDevice\\model\\refs\\' + url[i])) for i in range(len(url))]
        self.soundPlayers = [QMediaPlayer() for _ in range(len(url))]
        # for i in range(len(url)):
        #     self.soundPlayers[i].setMedia())

    def update_detetct_on_ON(self):
        self.detect_on = True

    def update_detetct_on_OFF(self):
        self.detect_on = False


    def run(self):
        start_image = QPixmap("./refs/background2.png")
        self.imgSignal.emit(start_image)

        self.cap = cv2.VideoCapture(0)
        self.image_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.image_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_3s = self.fps * 2
        self.alert_threshold = 2


        self.device = torch.device("cpu")
        self.onnx_model_path = './models/best.pt'
        self.scan_every = 1 * int(math.ceil((1.0 * self.fps / 10))) #设置检测的频率

        self.model = DetectMultiBackend(self.onnx_model_path, device=self.device, dnn=False, data='DriverFaceData.yaml', fp16=False)
        self.tracker = Tracker(self.image_width, self.image_height, scan_every=self.scan_every, silent=True, enable_debug=False, yolo_model=self.model, device=self.device, token=self.token, url=self.url)
        self.tracker.Init(self.image_width, self.image_height, self.scan_every, enable_gray=False)
        self.load_mp3()

        self.start_time = time.time()
        while self.running:
            if self.detect_on:
                ret, frame = self.cap.read()
                if ret:
                    #frame = cv2.resize(frame, (320, 320))
                    #start = time.time()
                    result_image, self.is_huawei_cloud_connected, delay = self.detect(frame)

                    height, width, channel = result_image.shape
                    bytes_per_line = channel * width
                    q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    q_image = q_image.rgbSwapped()

                    pixmap = QPixmap().fromImage(q_image)
                    self.imgSignal.emit(pixmap)
                    self.stateSignal.emit(self.is_huawei_cloud_connected)
                    self.delaySignal.emit(delay)
                    self.huawei_cloud_watcher_thread.setCloudState(self.is_huawei_cloud_connected) # 防止断网时监视器中的状态不改变
                    #scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                    #self.image_label.setPixmap(scaled_pixmap)
                    #self.image_label.setAlignment(Qt.AlignCenter)
            else:
                self.imgSignal.emit(start_image)
                time.sleep(0.3)

    
    def detect(self, frame):
        [close_eyes, yawn, on_call, look_left_right], results, isNetStateOk, delay = self.tracker.predict(frame, self.is_huawei_cloud_connected, '')
        
        #print("Request HUAWEI ModelArts Service OK." if isNetStateOk else "Fail request HUAWEI ModelArts Service.")

        if results is not None:
            self.bbox = results
        annotator = Annotator(frame, line_width=2, example=str(self.model.names))
        self.previous_frames_result[self.store_index][1: 5] = int(close_eyes), int(yawn), int(on_call), int(look_left_right)

        self.previous_frames_result_time[self.store_index] = time.time()

        for box in self.bbox:
            annotator.box_label([int(box[0]), int(box[1]), int(box[2]), int(box[3])], self.model.names[int(box[5])], color=colors(int(box[5]), True))

        for cls in [3, 4, 2, 1]:
            # self.presum[self.store_index][cls] = self.presum[self.store_index - 1][cls] + self.previous_frames_result[self.store_index][cls]
            # cur_frame_3s_sum = self.presum[self.store_index][cls] - self.presum[(int)(self.store_index - self.frame_3s + self.loop_array_length) % self.loop_array_length][cls]
            cur_frame_3s = 0
            tot_detect_frame = 0
            for j in range(self.loop_array_length):
                cur_index = (self.store_index - j + self.loop_array_length) % self.loop_array_length
                if self.previous_frames_result_time[self.store_index] - self.previous_frames_result_time[cur_index] < self.alert_threshold:
                    if self.previous_frames_result[cur_index][cls] == 1:
                        cur_frame_3s += 1
                    tot_detect_frame += 1
                else:
                    break


            if cur_frame_3s > 0.85 * tot_detect_frame and time.time() - self.start_time > self.alert_threshold:
                frame = self.image_margin_red(frame)
                try:
                    if self.sound_thread is None or not self.sound_thread.is_running():
                        #print("Start Sound, state = ", cls)
                        self.soundPlayers[cls - 1].setMedia(self.qMediaContents[cls - 1])
                        self.sound_thread = soundThread(self.soundPlayers[cls - 1])
                        self.sound_thread.start()
                        self.alert_thread = alertThread(self.verify_code, self.APIgateway, cls)
                        self.alert_thread.start()
                except Exception as e:
                    print('Error in create thread.')

        self.store_index = (self.store_index + 1) % self.loop_array_length
        return frame, isNetStateOk, delay

    def image_margin_red(self, img):
        blk = np.zeros(img.shape, np.uint8)
        for i in range(25):
            cv2.rectangle(blk, (10 * i, 10 * i), (img.shape[1] - 10 * i, img.shape[0] - 10 * i), (0, 0, 255 - 10 * i), i * 10 + 20)  # 注意在 blk的基础上进行绘制；
        picture = cv2.addWeighted(img, 0.8, blk, 1, 0)
        return picture

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.font_family = "华文中宋"

        # -------------------------------------------------------- 窗口布局 ----------------------------------------------------------

        # 创建一个主窗口部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建一个垂直布局，放入一个水平布局text_layout，一个image_label
        layout = QVBoxLayout()

        # 创建一个水平布局, 显示LOGO和标题
        text_layout = QHBoxLayout()
        self.huawei_logo = QLabel()
        self.desksize = QApplication.desktop().screenGeometry(0).width()
        #self.huawei_logo.setMaximumHeight(200)
        self.huawei_logo.setPixmap(QPixmap("./refs/huawei_logo.png").scaled(self.huawei_logo.size() * self.desksize / 3500, Qt.KeepAspectRatio))
        self.huawei_logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        text_layout.addWidget(self.huawei_logo)

        self.text_label = QLabel()
        self.text_label.setText("基于端云算力协同的疲劳驾驶智能识别系统") #基于端云算力协同的
        self.text_label.setAlignment((Qt.AlignCenter))
        self.text_label.setStyleSheet((f"font-size: {self.desksize // 30}px; color: black; font-family: {self.font_family}"))
        text_layout.addWidget((self.text_label))

        self.huake_logo = QLabel()
        #self.huake_logo.setMaximumHeight(200)
        self.huake_logo.setPixmap(QPixmap("./refs/huake_logo.png").scaled(self.huawei_logo.size() * self.desksize / 4300, Qt.KeepAspectRatio))
        self.huake_logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        text_layout.addWidget(self.huake_logo)
        layout.addLayout(text_layout, 1)

        state_and_pic_layout = QHBoxLayout()
        # 创建一个标签用于显示图片和状态栏
        # self.init_green_red_pixmap()
        self.state_bar = QVBoxLayout()
        # self.huawei_cloud_connect_state_label = QLabel() # 华为云连接状态
        # self.manage_connect_state_label = QLabel() # 管理后台链接状态
        self.huawei_cloud_layout = QHBoxLayout()
        self.huawei_cloud_connect_state_label = QLabel()
        self.huawei_cloud_connect_state_text = QLabel()
        self.huawei_cloud_layout.addWidget(self.huawei_cloud_connect_state_label, 1)
        self.huawei_cloud_layout.addWidget(self.huawei_cloud_connect_state_text, 4)
        self.huawei_cloud_connect_state_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        self.huawei_cloud_connect_state_text.setAlignment(Qt.AlignCenter | Qt.AlignLeft)

        self.manage_layout = QHBoxLayout()
        self.manage_connect_state_label = QLabel() # 管理后台链接状态
        self.manage_connect_state_text = QLabel()
        self.manage_layout.addWidget(self.manage_connect_state_label, 1)
        self.manage_layout.addWidget(self.manage_connect_state_text, 4)
        self.manage_connect_state_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        self.manage_connect_state_text.setAlignment(Qt.AlignCenter | Qt.AlignLeft)

        self.huawei_delay_layout = QHBoxLayout()
        self.huawei_cloud_delay_label = QLabel() # 华为云延迟
        self.huawei_cloud_delay_text = QLabel()
        self.huawei_delay_layout.addWidget(self.huawei_cloud_delay_label, 1)
        self.huawei_delay_layout.addWidget(self.huawei_cloud_delay_text, 2)
        self.huawei_cloud_delay_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        self.huawei_cloud_delay_text.setAlignment(Qt.AlignCenter | Qt.AlignLeft)

        # self.on_button.
        #switch.toggled.connect(self.switchChange)
        #self.state_and_pic_layout.addWidget(switch, 1)

        self.state_bar.addLayout(self.huawei_cloud_layout)
        self.state_bar.addLayout(self.manage_layout)
        self.state_bar.addLayout(self.huawei_delay_layout)
        #self.state_bar.addWidget(self.on_button)

        state_and_pic_layout.addLayout(self.state_bar, 1)

        self.image_label = QLabel()
        # self.image_label.setPixmap(self.start_image.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.image_label.setAlignment(Qt.AlignCenter)
        state_and_pic_layout.addWidget(self.image_label, 4)

        right_layout = QVBoxLayout()
        self.on_button = QPushButton('开始检测',self)
        self.on_button.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))
        right_layout.addWidget(self.on_button, 1)
        self.on_button.setMinimumHeight(100)

        # self.on_button.setAlignment(Qt.AlignCenter)

        self.off_button = QPushButton('关闭检测',self)
        self.off_button.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))
        right_layout.addWidget(self.off_button, 1)
        self.off_button.setMinimumHeight(100)
        #self.off_button.clicked.connect(self.off_button_func)
        # self.off_button.setAlignment(Qt.AlignCenter)

        self.log_bar = QLabel()
        right_layout.addWidget(self.log_bar, 3)


        state_and_pic_layout.addLayout(right_layout, 1)

        layout.addLayout(state_and_pic_layout, 4)

        # 设置窗口主布局
        central_widget.setLayout(layout)

        self.showMaximized()

        #--------------------------------------------------------------------------------------------------------------------------

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_image)
        # self.timer.start(30)

        # self.model_loaded = False

        # 告警信息后台
        self.is_manage_connected = False
        self.verificationCode = "DEVICE_0001" # 设备唯一标识码，管理员账户登记新设备时使用
        self.APIgateway = "http://114.116.213.82:44444"
        self.heart_beat_thread = heartbeatThread(self.verificationCode, self.APIgateway)
        self.heart_beat_thread.stateSignal.connect(self.update_manage_connect_state)
        self.heart_beat_thread.start()

        # 华为云连接状态监视
        self.is_huawei_cloud_connected = False
        self.token = "MIIRwQYJKoZIhvcNAQcCoIIRsjCCEa4CAQExDTALBglghkgBZQMEAgEwgg-TBgkqhkiG9w0BBwGggg-EBIIPwHsidG9rZW4iOnsiZXhwaXJlc19hdCI6IjIwMjMtMDktMTZUMTE6NTM6MTguMDk2MDAwWiIsIm1ldGhvZHMiOlsicGFzc3dvcmQiXSwiY2F0YWxvZyI6W10sInJvbGVzIjpbeyJuYW1lIjoidGVfYWRtaW4iLCJpZCI6IjAifSx7Im5hbWUiOiJ0ZV9hZ2VuY3kiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3JlcF9hY2NlbGVyYXRpb24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfZGlza0FjYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Rzc19tb250aCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29ic19kZWVwX2FyY2hpdmUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfc3BvdF9pbnN0YW5jZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Jkc19tYXJpYWRiX29idCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Jjc19uZXNfc2ciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLXNvdXRoLTRjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfUlRTQV9jb25jdXJyZW50X2NoYW5uZWwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9kZWNfbW9udGhfdXNlciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2ludGxfb2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jYnJfc2VsbG91dCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Zsb3dfY2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfb2xkX3Jlb3VyY2UiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9wYW5ndSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3dlbGlua2JyaWRnZV9lbmRwb2ludF9idXkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jYnJfZmlsZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2lkbWVfbGlua3hfZm91bmRhdGlvbiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Rtcy1yb2NrZXRtcTUtYmFzaWMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9kbXMta2Fma2EzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc250OWJpbmwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lZGdlc2VjX29idCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29ic19kdWFsc3RhY2siLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vYnNfZGVjX21vbnRoIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3Nic19yZXN0b3JlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX2M2YSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX0VDX09CVCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2tvb3Bob25lIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfbXVsdGlfYmluZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Ntbl9jYWxsbm90aWZ5IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3JnaWRfY2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2FwLXNvdXRoZWFzdC0zZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2lhbV9pZGVudGl0eWNlbnRlcl9pbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3Nic19wcm9ncmVzc2JhciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Nlc19yZXNvdXJjZWdyb3VwX3RhZyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19vZmZsaW5lX2FjNyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2V2c19yZXR5cGUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9rb29tYXAiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9ldnNfZXNzZDIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9ldnNfcG9vbF9jYSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3BlZGFfc2NoX2NhIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9jbi1zb3V0aHdlc3QtMmIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9od2NwaCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19vZmZsaW5lX2Rpc2tfNCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2h3ZGV2IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3BfZ2F0ZWRfY2JoX3ZvbHVtZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Ntbl93ZWxpbmtyZWQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9kYXRhYXJ0c2luc2lnaHQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9odl92ZW5kb3IiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF91Y3Nfb25fYXdzX2ludGwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLW5vcnRoLTRlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfd2FmX2NtYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfY24tbm9ydGgtNGQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfYWM3IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZXBzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3Nic19yZXN0b3JlX2FsbCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29yZ2FuaXphdGlvbnNfaW50bCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Jkc19tYXJpYWRiX29idF9JbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWRzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaWFtX2lkZW50aXR5Y2VudGVyX2NuIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9jbi1ub3J0aC00ZiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29hIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc2ZzX2xpZmVjeWNsZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX29wX2dhdGVkX3JvdW5kdGFibGUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2FwLXNvdXRoZWFzdC0xZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfcnUtbW9zY293LTFiIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9hcC1zb3V0aGVhc3QtMWQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2FwLXNvdXRoZWFzdC0xZiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Ntbl9hcHBsaWNhdGlvbiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NzZV9nYXRld2F5IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc250OWJpNmwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9JUERDZW50ZXJfQ0FfMjAyMzA4MzAiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9yYW0iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vcmdhbml6YXRpb25zIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX2dwdV9nNXIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vcF9nYXRlZF9tZXNzYWdlb3ZlcjVnIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfcmlfZHdzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX3JpIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfbWdjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc250OWIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX3J1LW5vcnRod2VzdC0yYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3JhbV9pbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaWVmX3BsYXRpbnVtIiwiaWQiOiIwIn1dLCJwcm9qZWN0Ijp7ImRvbWFpbiI6eyJuYW1lIjoiY2htb2QtNzc3IiwiaWQiOiI1MDYzM2ZkZjY1N2Q0NDdmYjQzNzAxMjczNzgxYzE2OSJ9LCJuYW1lIjoiY24tbm9ydGgtNCIsImlkIjoiM2RlNmUxNDlkNTExNGMxN2FlZWUxNGZlODhhYmNlY2EifSwiaXNzdWVkX2F0IjoiMjAyMy0wOS0xNVQxMTo1MzoxOC4wOTYwMDBaIiwidXNlciI6eyJkb21haW4iOnsibmFtZSI6ImNobW9kLTc3NyIsImlkIjoiNTA2MzNmZGY2NTdkNDQ3ZmI0MzcwMTI3Mzc4MWMxNjkifSwibmFtZSI6ImNobW9kLTY2NiIsInBhc3N3b3JkX2V4cGlyZXNfYXQiOiIiLCJpZCI6ImM1ODc3ODk2NmY2YjQzZTdiN2E4OGQwNDJmOGY4NWRmIn19fTGCAcEwggG9AgEBMIGXMIGJMQswCQYDVQQGEwJDTjESMBAGA1UECAwJR3VhbmdEb25nMREwDwYDVQQHDAhTaGVuWmhlbjEuMCwGA1UECgwlSHVhd2VpIFNvZnR3YXJlIFRlY2hub2xvZ2llcyBDby4sIEx0ZDEOMAwGA1UECwwFQ2xvdWQxEzARBgNVBAMMCmNhLmlhbS5wa2kCCQDcsytdEGFqEDALBglghkgBZQMEAgEwDQYJKoZIhvcNAQEBBQAEggEAdZQ0Zx5YgCis3NpAbKPrt6qrux-fUBYXC9E4fD1dF1-m7zC+0URvkGJAg2MG0f7CbpfSMo+cZEYC1Mweiqmic-7Q1ogMXKjf1L8cdJ+fG+RXdlR1mV3yWUQObq6AhPHQ0SK7WYG05lKirt-H9EnbMkOuFPO2q+xot48cniWeyR1-pkYucrI+1p0EynBr1eWvedr+ggBh+FgW-iJ087HuDCzyLAxfq9oGs1dLRC1i8N3vaw3y2jTM5HZiLHqTGSPONb-36MslQFTNoj23sGcEogl0GhFFN6pbQ3Y+KJYJ5rbojFK8UNlkWncOL7LgMpZ46JP83ilReJFIFw6ho3mtZA=="
        self.url = "https://e7ace0dd9b76405abede816865e9babb.apigw.cn-north-4.huaweicloud.com/v1/infers/41a1444e-b379-4a74-bfda-0307d338d235" # 在线服务URL
        # self.huawei_cloud_watcher_thread = huaweiCloudWatcherThread(self.token, self.url)
        # self.huawei_cloud_watcher_thread.stateSignal.connect(self.update_huawei_cloud_connect_state)
        # self.huawei_cloud_watcher_thread.start()

        # 推理服务
        self.huawei_cloud_delay = 0
        self.inference_thread = updateImageThread(self.token, self.url, self.verificationCode, self.APIgateway)
        self.inference_thread.imgSignal.connect(self.update_image)
        self.inference_thread.stateSignal.connect(self.update_huawei_cloud_connect_state)
        self.inference_thread.delaySignal.connect(self.update_huawei_delay)
        self.inference_thread.start()

        self.on_button.clicked.connect(self.inference_thread.update_detetct_on_ON)
        self.off_button.clicked.connect(self.inference_thread.update_detetct_on_OFF)

        self.setWindowTitle("演示Demo")

        print("Load model finished!")

    def on_button_func(self):
        self.inference_thread.update_detetct_on(True)

    def off_button_func(self):
        self.inference_thread.update_detetct_off(False)

    def update_start_image(self):
        self.image_label.setPixmap(self.start_image.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.image_label.setAlignment(Qt.AlignCenter)

    def init_green_red_pixmap(self):
        self.red_pixmap = self.green_pixmap = QPixmap(50, 50)
        self.red_pixmap.fill(Qt.transparent)
        self.green_pixmap.fill(Qt.transparent)

        with QPainter(self.red_pixmap) as painter:
            red_color = QColor(255, 0, 0)
            painter.setBrush(red_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 50, 50)

        with QPainter(self.green_pixmap) as painter:
            green_color = QColor(0, 255, 0)
            painter.setBrush(green_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 50, 50)

    def draw_red_ellipse(self):
        red_pixmap = QPixmap(50, 50)
        red_pixmap.fill(Qt.transparent)
        with QPainter(red_pixmap) as painter:
            red_color = QColor(255, 0, 0)
            painter.setBrush(red_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 50, 50)
            return red_pixmap

    def draw_green_ellipse(self):
        green_pixmap = QPixmap(50, 50)
        green_pixmap.fill(Qt.transparent)
        with QPainter(green_pixmap) as painter:
            green_color = QColor(0, 255, 0)
            painter.setBrush(green_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 50, 50)
            return green_pixmap

    def update_state_bar(self):
        #print("update_state_bar: ", self.is_huawei_cloud_connected, self.is_manage_connected)
        huawei_cloud_pixmap = self.draw_green_ellipse() if self.is_huawei_cloud_connected else self.draw_red_ellipse()
        manage_pixmap = self.draw_green_ellipse() if self.is_manage_connected else self.draw_red_ellipse()

        self.huawei_cloud_connect_state_label.setPixmap(huawei_cloud_pixmap)
        self.huawei_cloud_connect_state_text.setText(" 华为云连接正常" if self.is_huawei_cloud_connected else " 华为云连接异常")
        self.huawei_cloud_connect_state_text.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))
        #self.text_label.setStyleSheet(("font-size: 150px; color: black; font-family: 楷体"))
        self.manage_connect_state_label.setPixmap(manage_pixmap)
        self.manage_connect_state_text.setText(" 管理平台连接正常" if self.is_manage_connected else " 管理平台连接异常")
        self.manage_connect_state_text.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))
        
        self.huawei_cloud_delay_label.setText("华为云请求延迟: ")
        self.huawei_cloud_delay_text.setText(f" {self.huawei_cloud_delay} ms" if self.is_huawei_cloud_connected else " 连接异常")
        self.huawei_cloud_delay_label.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))
        self.huawei_cloud_delay_text.setStyleSheet((f"font-size: {self.desksize // 70}px; color: black; font-family: {self.font_family}"))

    def update_huawei_delay(self, delay):
        self.huawei_cloud_delay = delay

    def update_manage_connect_state(self, is_manage_connected):
        self.is_manage_connected = is_manage_connected

    def update_huawei_cloud_connect_state(self, is_huawei_cloud_connected):
        self.is_huawei_cloud_connected = is_huawei_cloud_connected
    
    def load_mp3(self):
        url = ['closed_eyes.mp3', 'yawn.mp3', 'on_call.mp3', 'look_left_right.mp3']
        self.soundPlayers = [QMediaPlayer() for _ in range(len(url))]
        for i in range(len(url)):
            self.soundPlayers[i].setMedia(QMediaContent(QUrl.fromLocalFile('./refs/' + url[i])))

    def load_model(self):
        # 加载模型
        self.model = DetectMultiBackend(self.onnx_model_path, device=self.device, dnn=False, data='DriverFaceData.yaml', fp16=False)
        self.tracker = Tracker(self.image_width, self.image_height, scan_every=1, silent=True, enable_debug=False, yolo_model=self.model, device=self.device)
        self.tracker.Init(self.image_width, self.image_height, self.scan_every, enable_gray=False)
        self.load_mp3()
        self.model_loaded = True

    def image_margin_red(self, img):
        blk = np.zeros(img.shape, np.uint8)
        for i in range(25):
            cv2.rectangle(blk, (10 * i, 10 * i), (img.shape[1] - 10 * i, img.shape[0] - 10 * i), (0, 0, 255 - 10 * i), i * 10 + 20)  # 注意在 blk的基础上进行绘制；
        picture = cv2.addWeighted(img, 0.8, blk, 1, 0)
        return picture

    def update_image(self, pixmap):
        # print("update_image: ", self.model_loaded)
        # if self.model_loaded is True:
        #     ret, frame = self.cap.read()
        #     if ret:

        #         result_image = self.detect(frame)

        #         height, width, channel = result_image.shape
        #         bytes_per_line = channel * width
        #         q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        #         q_image = q_image.rgbSwapped()

        #         pixmap = QPixmap().fromImage(q_image)
        #         scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        #         self.image_label.setPixmap(scaled_pixmap)
        #         self.image_label.setAlignment(Qt.AlignCenter)

        # else:
        #     scaled_pixmap = self.start_image.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        #     self.image_label.setPixmap(scaled_pixmap)
        #     self.image_label.setAlignment(Qt.AlignCenter)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.update_state_bar()

    # def update_watcher_state(self, new_state):
    #     self.huawei_cloud_watcher_thread.setCloudState(new_state)

    def heartbeat_thread(self):
        while True:
            ret = requests.put("{}/device/heartbeat?verificationCode={}".format(self.APIgateway, self.verificationCode))
            print('heartbeat')
            print("answer for heartbeat: " + str(ret.json()))
            self.is_manage_connected = (200 <= ret.status_code <= 299)
            time.sleep(60)  # 心跳周期1min


    # 向管理平台发布告警通知
    def alert(self, type):
        params = {
            "deviceVerificationCode": self.verificationCode,
            "type": type,
            # TODO 添加用户名、联系方式
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        }
        print("alert type-" + str(type))
        ret = requests.post("{}/alert/".format(self.APIgateway), data=params)
        self.is_manage_connected = (200 <= ret.status_code <= 299)
        print("answer for alert submission: " + str(ret.json()))

    # def closeEvent(self, a0: QCloseEvent) -> None:
    #     self.heart_beat_thread.stop()
    #     #self.heart_beat_thread.wait()
    #     self.huawei_cloud_watcher_thread.stop()
    #     #self.huawei_cloud_watcher_thread.wait()
    #     self.inference_thread.stop()
    #     #self.inference_thread.wait()

    #     return super().closeEvent(a0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.setWindowIcon(QIcon("./refs/huawei_logo.png"))

    # window.load_model()
    # load_model_thread = threading.Thread(target=window.load_model)
    # load_model_thread.start()

    window.show()
    sys.exit(app.exec_())