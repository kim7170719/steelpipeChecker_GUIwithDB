import sys
import os

# 判斷是否為PyInstaller打包環境
if getattr(sys, 'frozen', False):
    # 打包的路徑
    base_path = sys._MEIPASS
else:
    # 開發路徑
    base_path = os.path.dirname(os.path.abspath(__file__))

# 加載字體、圖片、模型的代碼示例
font_path = os.path.join(base_path, 'msjh.ttc')
logo_path = os.path.join(base_path, 'nfu_logo.png')
model_path = os.path.join(base_path, 'best.pt')
import requests
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle
from kivy.uix.scrollview import ScrollView
import cv2
import numpy as np
import math
from collections import defaultdict
import statistics
from datetime import datetime

# 導入 6.0.py 中的函數
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detect_chessboard_calibration(image, pattern_size=(9, 9), square_size_cm=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    distances = []
    num_cols = pattern_size[0]
    for i in range(pattern_size[1]):
        for j in range(num_cols - 1):
            idx1 = i * num_cols + j
            idx2 = i * num_cols + j + 1
            pt1 = corners[idx1, 0]
            pt2 = corners[idx2, 0]
            distances.append(np.linalg.norm(pt2 - pt1))
    avg_distance = np.mean(distances)
    pixels_per_cm = avg_distance / square_size_cm

    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    return pixels_per_cm, corners

def detect_inner_circle_advanced(roi, outer_circle):
    cx_o, cy_o, r_o = outer_circle
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 多種二值化方法
    methods = [
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_BINARY_INV
    ]
    
    best_circle = None
    best_score = float('-inf')
    
    for method in methods:
        if method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_BINARY_INV:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, binary = cv2.threshold(gray, 0, 255, method)
        
        # 形態學處理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)
            
            # 更嚴格的篩選條件
            if radius < 5 or radius > r_o * 0.8:
                continue
            
            d = distance((x, y), (cx_o, cy_o))
            
            # 計算圓形度
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 綜合評分
            score = circularity - (d / r_o)
            
            if score > best_score:
                best_score = score
                best_circle = (x, y, radius)
    
    return best_circle

# 加入背景元件類別
class BackgroundLabel(Label):
    def __init__(self, **kwargs):
        super(BackgroundLabel, self).__init__(**kwargs)
        self.padding = [15, 15]
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 0.8)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

# 標題元件類別
class TitleBar(BoxLayout):
    def __init__(self, **kwargs):
        super(TitleBar, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint = (1, 0.1)
        self.padding = [10, 5]
        self.spacing = 10
        self.font_name = 'msjh.ttc'  # 統一使用註冊字型名稱
        with self.canvas.before:
            Color(0.1, 0.1, 0.3, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        # Logo 和標題
        logo_layout = BoxLayout(orientation='vertical', size_hint=(0.2, 1))
        self.logo = Image(source='', size_hint=(1, 0.7))
        try:
            if os.path.exists('nfu_logo.png'):
                self.logo.source = 'nfu_logo.png'
        except:
            pass
        logo_layout.add_widget(self.logo)
        
        # 標題區
        title_layout = BoxLayout(orientation='vertical', size_hint=(0.8, 1))
        self.title = Label(
            text='NFU 虎尾科技大學電子工程系',
            font_size='20sp',
            halign='left',
            font_name=self.font_name,  # 使用註冊字型名稱
            bold=True
        )
        self.subtitle = Label(
            text='智慧型視覺運算實驗室  Intelligent Vision Computing Laboratory',
            font_size='16sp',
            halign='left',
            font_name=self.font_name  # 使用註冊字型名稱
        )
        title_layout.add_widget(self.title)
        title_layout.add_widget(self.subtitle)
        
        self.add_widget(logo_layout)
        self.add_widget(title_layout)
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

# 現代化按鈕樣式
class ModernButton(Button):
    def __init__(self, **kwargs):
        super(ModernButton, self).__init__(**kwargs)
        self.background_color = (0.2, 0.4, 0.8, 1)
        self.background_normal = ''
        self.bold = True
        self.border = (5, 5, 5, 5)

class SteelPipeDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.orientation = 'vertical'
        self.padding = [10, 10]
        self.spacing = 10
        self.detection_text = None  # 儲存檢測文本
        self.detection_result_img = None  # 儲存檢測結果圖像
        ctrl_layout = BoxLayout(size_hint=(1, 0.1), spacing=10, padding=10)
        # 背景
        with self.canvas.before:
            Color(0.15, 0.15, 0.2, 1)  # 深色背景
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.pipe_image_path = None
        self.grid_image_path = None
        self.calibration_value = None
        self.font_path = 'msjh.ttc'  # 字型檔須放在.py同目錄下
        
        # 理論值設定
        self.THEORY_OUTER = [20, 12, 10]
        self.THEORY_INNER_MAP = {
            20: [10, 15, 12, 14],
            12: [6, 7],
            10: [6]
        }
        
        # 添加標題欄
        self.title_bar = TitleBar()
        self.add_widget(self.title_bar)
        
        # 檔案選擇部分
        file_selection_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.3))
        
        # 棋盤格圖片選擇區域
        self.grid_section = BoxLayout(orientation='vertical')
        self.grid_image = Image(source='', size_hint=(1, 0.8), allow_stretch=True)
        grid_btn = ModernButton(
            text='選擇棋盤格圖片', 
            font_name=self.font_path, 
            size_hint=(1, 0.2),
            on_press=self.show_grid_chooser
        )
        self.grid_section.add_widget(grid_btn)
        self.grid_section.add_widget(self.grid_image)
        
        # 鋼管圖片選擇區域
        self.pipe_section = BoxLayout(orientation='vertical')
        self.pipe_image = Image(source='', size_hint=(1, 0.8), allow_stretch=True)
        pipe_btn = ModernButton(
            text='選擇鋼管圖片', 
            font_name=self.font_path, 
            size_hint=(1, 0.2),
            on_press=self.show_pipe_chooser
        )
        self.pipe_section.add_widget(pipe_btn)
        self.pipe_section.add_widget(self.pipe_image)
        
        file_selection_layout.add_widget(self.grid_section)
        file_selection_layout.add_widget(self.pipe_section)
        
        # 參數設定部分
        param_layout = GridLayout(cols=2, size_hint=(1, 0.2), spacing=10, padding=10)
        param_layout.bind(minimum_height=param_layout.setter('height'))
        
        # 棋盤格行數
        param_layout.add_widget(BackgroundLabel(
            text='棋盤格行數:',
            font_name=self.font_path,
            halign='right'
        ))
        self.rows_input = TextInput(text='9', input_filter='int', multiline=False)
        param_layout.add_widget(self.rows_input)
        
        # 棋盤格列數
        param_layout.add_widget(BackgroundLabel(
            text='棋盤格列數:',
            font_name=self.font_path,
            halign='right'
        ))
        self.cols_input = TextInput(text='9', input_filter='int', multiline=False)
        param_layout.add_widget(self.cols_input)
        
        # 棋盤格方塊尺寸(公分)
        param_layout.add_widget(BackgroundLabel(
            text='方塊尺寸(公分):',
            font_name=self.font_path,
            halign='right'
        ))
        self.square_size_input = TextInput(text='0.5', input_filter='float', multiline=False)
        param_layout.add_widget(self.square_size_input)
        
        # 控制按鈕部分
        ctrl_layout = BoxLayout(size_hint=(1, 0.1), spacing=10, padding=10)
        
        self.calibrate_btn = ModernButton(
            text='進行校正', 
            font_name=self.font_path,
            background_color=(0.2, 0.6, 0.2, 1)
        )
        self.calibrate_btn.bind(on_press=self.calibrate)
        
        self.detect_btn = ModernButton(
            text='檢測鋼管', 
            font_name=self.font_path,
            background_color=(0.2, 0.2, 0.8, 1)
        )
        self.detect_btn.bind(on_press=self.detect_pipes)
        
        self.save_btn = ModernButton(
            text='儲存結果', 
            font_name=self.font_path, 
            disabled=True,
            background_color=(0.8, 0.4, 0.2, 1)
        )
        self.save_btn.bind(on_press=self.save_result)
        
        self.sync_btn = ModernButton(
            text='同步至雲端資料庫',
            font_name=self.font_path,
            disabled=True,
            background_color=(0.4, 0.2, 0.6, 1)
        )
        # 綁定按下事件到 sync_to_cloud 函數
        self.sync_btn.bind(on_press=self.sync_to_cloud)
        
        ctrl_layout.add_widget(self.calibrate_btn)
        ctrl_layout.add_widget(self.detect_btn)
        ctrl_layout.add_widget(self.save_btn)
        ctrl_layout.add_widget(self.sync_btn)
        # 狀態顯示
        self.status_layout = BoxLayout(size_hint=(1, 0.05))
        with self.status_layout.canvas.before:
            Color(0.2, 0.2, 0.2, 0.8)
            self.status_rect = Rectangle(size=self.status_layout.size, pos=self.status_layout.pos)
        self.status_layout.bind(size=self._update_status_rect, pos=self._update_status_rect)
        
        self.status_label = Label(
            text='請選擇圖片並設定參數', 
            font_name=self.font_path, 
            size_hint=(1, 1)
        )
        self.status_layout.add_widget(self.status_label)
        
        # 輸出圖像
        self.output_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.4))
        self.output_image = Image(source='', size_hint=(1, 1), allow_stretch=True)
        self.output_layout.add_widget(self.output_image)
        
        # 添加所有元件到主布局
        self.add_widget(file_selection_layout)
        self.add_widget(param_layout)
        self.add_widget(ctrl_layout)
        self.add_widget(self.status_layout)
        self.add_widget(self.output_layout)
        
        # 檢測結果
        self.detection_result = None
        
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        
    def _update_status_rect(self, instance, value):
        self.status_rect.pos = instance.pos
        self.status_rect.size = instance.size

    def show_pipe_chooser(self, instance):
        self.show_file_chooser('pipe')

    def show_grid_chooser(self, instance):
        self.show_file_chooser('grid')

    def show_file_chooser(self, img_type):
        content = BoxLayout(orientation='vertical')
        # 只顯示圖片檔案，並指定中文字型
        chooser = FileChooserIconView(
            filters=['*.png', '*.jpg', '*.jpeg', '*.bmp'],
            font_name=self.font_path
        )
        
        select_btn = ModernButton(
            text='選擇',
            font_name=self.font_path,
            size_hint_y=None,
            height=50
        )
        cancel_btn = ModernButton(
            text='取消',
            font_name=self.font_path,
            size_hint_y=None,
            height=50
        )
        
        buttons = BoxLayout(size_hint_y=None, height=50)
        buttons.add_widget(select_btn)
        buttons.add_widget(cancel_btn)
        
        content.add_widget(chooser)
        content.add_widget(buttons)
        
        # 指定標題字型，避免標題亂碼
        popup = Popup(
            title='選擇圖片',
            title_font=self.font_path,
            content=content,
            size_hint=(0.9, 0.9)
        )

        def select_image(instance):
            if chooser.selection:
                selected = chooser.selection[0]
                if img_type == 'pipe':
                    self.pipe_image_path = selected
                    self.pipe_image.source = selected
                    self.pipe_image.reload()
                elif img_type == 'grid':
                    self.grid_image_path = selected
                    self.grid_image.source = selected
                    self.grid_image.reload()
                popup.dismiss()

        select_btn.bind(on_press=select_image)
        cancel_btn.bind(on_press=popup.dismiss)
        chooser.bind(on_submit=lambda chooser, selection, touch: select_image(None))

        popup.open()

    
    def calibrate(self, instance):
        if not self.grid_image_path:
            self.show_error("請先選擇棋盤格圖片！")
            return
            
        try:
            # 讀取參數
            rows = int(self.rows_input.text)
            cols = int(self.cols_input.text)
            square_size = float(self.square_size_input.text)
            
            # 讀取圖片
            grid_img = cv2.imread(self.grid_image_path)
            
            # 進行校正
            pixels_per_cm, corners = detect_chessboard_calibration(
                grid_img, pattern_size=(cols, rows), square_size_cm=square_size
            )
            
            if pixels_per_cm is None:
                self.show_error("找不到棋盤格！請檢查參數是否正確。")
                return
                
            # 儲存校正係數
            self.calibration_value = pixels_per_cm
            
            # 顯示校正結果
            display_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            display_img = cv2.flip(display_img, 0) 
            self.show_processed_image(display_img)
            self.status_label.text = f"校正完成: {pixels_per_cm:.2f} 像素/公分"
            
            # 也保存到文件中
            with open("calibration_value.txt", "w") as f:
                f.write(str(pixels_per_cm))
                
        except Exception as e:
            self.show_error(f"校正失敗: {str(e)}")
    
    def detect_pipes(self, instance):
        if not self.pipe_image_path:
            self.show_error("請先選擇鋼管圖片！")
            return
            
        if not self.calibration_value and os.path.exists("calibration_value.txt"):
            try:
                with open("calibration_value.txt", "r") as f:
                    self.calibration_value = float(f.read().strip())
            except:
                pass
                
        if not self.calibration_value:
            self.show_error("請先進行校正！")
            return
            
        try:
            # 讀取鋼管圖片
            orig_img = cv2.imread(self.pipe_image_path)
            
            # 修正圖片顛倒問題
            #orig_img = cv2.rotate(orig_img, cv2.ROTATE_360)
            
            # 調整大小
            TARGET_SIZE = (1560, 1170)
            orig_h, orig_w = orig_img.shape[:2]
            
            scale = min(TARGET_SIZE[0]/orig_w, TARGET_SIZE[1]/orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_img = cv2.resize(orig_img, (new_w, new_h))
            
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            
            # 尋找 YOLOv8 模型
            model_paths = ["yolov8-steel-pipe model.pt", "yolov8-steel-pipe.pt", "steel-pipe-model.pt", "best.pt"]
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                self.show_error("錯誤: 找不到模型檔案")
                return
            
            # 使用模型檢測
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                results = model.predict(resized_img, conf=0.5, imgsz=640)
                boxes = results[0].boxes.xyxy.cpu().numpy()
            except Exception as e:
                self.show_error(f"模型執行失敗: {e}")
                return
                
            display_img = orig_img.copy()
            pipe_id = 0
            
            # 存儲統計數據的字典
            size_stats = defaultdict(int)
            
            # 檢測結果字串
            detection_text = "="*65 + "\n"
            detection_text += f"{'鋼管編號':<20}{'外徑(mm)':<25}{'內徑(mm)':<25}{'壁厚(mm)':<25}\n"
            detection_text += "-"*119 + "\n"
            
            for box in boxes:
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                roi = orig_img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                pipe_id += 1
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r_outer = (x2 - x1) // 2
                
                outer_circle_roi = (roi.shape[1]//2, roi.shape[0]//2, r_outer)
                inner_circle = detect_inner_circle_advanced(roi, outer_circle_roi)
                
                if inner_circle:
                    ix, iy, ir = inner_circle
                    abs_ix = x1 + ix
                    abs_iy = y1 + iy
                else:
                    ir = int(r_outer * 0.7)
                    abs_ix = cx
                    abs_iy = cy
                
                outer_diameter_px = 2 * r_outer
                inner_diameter_px = 2 * ir
                outer_diameter_mm = (outer_diameter_px / self.calibration_value) * 10
                inner_diameter_mm = (inner_diameter_px / self.calibration_value) * 10
                
                # 理論值修正處理
                corrected_outer = min(self.THEORY_OUTER, key=lambda x: abs(x - outer_diameter_mm))
                inner_candidates = self.THEORY_INNER_MAP.get(corrected_outer, [])
                if inner_candidates:
                    corrected_inner = min(inner_candidates, key=lambda x: abs(x - inner_diameter_mm))
                else:
                    corrected_inner = inner_diameter_mm

                outer_diameter_mm = corrected_outer
                inner_diameter_mm = corrected_inner
                wall_thickness = (outer_diameter_mm - inner_diameter_mm) / 2
                
                # 記錄尺寸統計
                size_key = (outer_diameter_mm, inner_diameter_mm)
                size_stats[size_key] += 1
                
                cv2.circle(display_img, (cx, cy), r_outer, (255, 0, 0), 4)
                cv2.circle(display_img, (abs_ix, abs_iy), ir, (0, 0, 255), 4)
                cv2.putText(display_img, f"Pipe {pipe_id}", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)

                detection_text += f"#{pipe_id:<30}{outer_diameter_mm:<35.1f}{inner_diameter_mm:<35.1f}{wall_thickness:<35.1f}\n"
            
            detection_text += "="*65 + "\n\n"
            
            # 尺寸統計
            detection_text += "鋼管尺寸統計表：\n"
            detection_text += "="*35 + "\n"
            detection_text += f"{'外徑(mm)':<15}{'內徑(mm)':<20}{'數量':<20}\n"
            detection_text += "-"*64 + "\n"
            
            # 按外徑排序後再按內徑排序
            for (od, id), count in sorted(sorted(size_stats.items(), key=lambda x: x[0][1]), key=lambda x: x[0][0]):
                detection_text += f"{od:<30}{id:<25}{count:<20}\n"
            
            detection_text += "="*35 + "\n"
            
            # 保存檢測結果
            self.detection_result = display_img
            self.detection_text = detection_text
            self.detection_result_img = display_img.copy()
            # 顯示處理後的圖片
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            display_rgb = cv2.flip(display_rgb, 0)
            self.show_processed_image(display_rgb)
            
            # 顯示檢測結果
            self.status_label.text = f"檢測完成：共發現 {pipe_id} 根鋼管"
            self.save_btn.disabled = False
            self.sync_btn.disabled = False
            
            # 顯示詳細結果
            self.show_detection_result(detection_text)
            
        except Exception as e:
            self.show_error(f"檢測失敗: {str(e)}")
    
    def show_processed_image(self, mat):
        buf = mat.tobytes()
        texture = Texture.create(
            size=(mat.shape[1], mat.shape[0]), 
            colorfmt='rgb'
        )
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.output_image.texture = texture

    def save_result(self, instance):
        if self.detection_result is not None:
            self.choose_save_location()
        else:
            self.show_error("沒有結果可以儲存！")
        
    def sync_to_cloud(self, instance):
            # 確保已選擇圖片並完成檢測
            if self.grid_image_path is None or self.pipe_image_path is None or self.detection_result_img is None:
                self.show_error("請先完成檢測！")
                return

            try:
                # 構建上傳資料
                upload_data = {
                    'grid_rows': self.rows_input.text,
                    'grid_cols': self.cols_input.text,
                    'square_size': self.square_size_input.text,
                    'pixels_per_cm': self.calibration_value,
                    'result_text': self.detection_text
                }

                from io import BytesIO
                _, buffer = cv2.imencode('.jpg', self.detection_result_img)
                output_bytes = BytesIO(buffer)

                response = requests.post(
                    'http://localhost:5000/api/static/uploads',
                    data=upload_data,
                    files={
                        'grid_image': open(self.grid_image_path, 'rb'),
                        'pipe_image': open(self.pipe_image_path, 'rb'),
                        'output_image': ('output.jpg', output_bytes.getvalue())
                    }
                )

                if response.status_code == 200:
                    self.show_popup("同步成功", "資料已同步至雲端資料庫")
                else:
                    self.show_error(f"伺服器錯誤({response.status_code}): {response.text}")

            except FileNotFoundError as e:
                self.show_error(f"檔案不存在: {str(e)}")
            except requests.exceptions.ConnectionError:
                self.show_error("無法連接伺服器，請檢查網路連線")
            except Exception as e:
                self.show_error(f"同步失敗: {str(e)}")
    def choose_save_location(self, text_content=None):
        content = BoxLayout(orientation='vertical')
        
        # 預設檔名
        now = datetime.now()
        default_filename = f"pipe_detection_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # 檔案選擇器，指定中文字型
        file_chooser = FileChooserIconView(
            dirselect=True,
            font_name=self.font_path
        )
        
        # 檔名輸入區
        filename_layout = BoxLayout(size_hint_y=None, height=40)
        filename_layout.add_widget(Label(
            text='檔案名稱:',
            size_hint_x=0.3,
            font_name=self.font_path
        ))
        filename_input = TextInput(
            text=default_filename,
            multiline=False,
            size_hint_x=0.7,
            font_name=self.font_path
        )
        filename_layout.add_widget(filename_input)
        
        # 按鈕區，指定中文字型
        btn_layout = BoxLayout(size_hint_y=None, height=50)
        save_btn = ModernButton(
            text='儲存',
            font_name=self.font_path
        )
        cancel_btn = ModernButton(
            text='取消',
            font_name=self.font_path
        )
        btn_layout.add_widget(save_btn)
        btn_layout.add_widget(cancel_btn)
        
        # 標題文字也要指定中文字型
        content.add_widget(Label(
            text='選擇儲存位置:',
            font_name=self.font_path
        ))
        content.add_widget(file_chooser)
        content.add_widget(filename_layout)
        content.add_widget(btn_layout)
        
        # Popup 加上 title_font
        popup = Popup(
            title='儲存檢測結果',
            title_font=self.font_path,
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def on_save(instance):
            if file_chooser.selection:
                directory = file_chooser.selection[0]
                if os.path.isdir(directory):
                    filepath = os.path.join(directory, filename_input.text)
                    try:
                        cv2.imwrite(filepath, self.detection_result)
                        self.show_popup("儲存成功", f"結果已儲存為 {filepath}")
                        popup.dismiss()
                    except Exception as e:
                        self.show_error(f"儲存失敗: {str(e)}")
                else:
                    self.show_error("請選擇有效的目錄")
            else:
                self.show_error("請選擇儲存位置")
        
        save_btn.bind(on_press=on_save)
        cancel_btn.bind(on_press=popup.dismiss)
        
        popup.open()


    def show_error(self, message):
        content = BoxLayout(orientation='vertical')
        error_label = Label(text=message, font_name=self.font_path)
        ok_btn = ModernButton(text='確定', size_hint=(None, None), size=(100, 50),font_name=self.font_path, pos_hint={'center_x': 0.5})
        content.add_widget(error_label)
        content.add_widget(ok_btn)
        
        popup = Popup(title='錯誤', content=content, size_hint=(0.8, 0.3),title_font=self.font_path, auto_dismiss=False)
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical')
        msg_label = Label(text=message, font_name=self.font_path)
        ok_btn = ModernButton(text='確定', size_hint=(None, None), size=(100, 50),font_name=self.font_path, pos_hint={'center_x': 0.5})
        content.add_widget(msg_label)
        content.add_widget(ok_btn)
        content = BoxLayout(orientation='vertical')
        msg_label = Label(text=message, font_name=self.font_path)
        ok_btn = ModernButton(text='確定', size_hint=(None, None), size=(100, 50),font_name=self.font_path, pos_hint={'center_x': 0.5})
        content.add_widget(msg_label)
        content.add_widget(ok_btn)
        
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.3), title_font=self.font_path, auto_dismiss=False)
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()
        
    def show_detection_result(self, text):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # 設置讀取文字的滾動視窗
        scroll_view = ScrollView(size_hint=(1, 0.9))
        results_text = TextInput(
            text=text, 
            readonly=True, 
            font_name=self.font_path, 
            size_hint=(1, None),
            background_color=(0.2, 0.2, 0.25, 1),
            foreground_color=(1, 1, 1, 1)
        )
        results_text.bind(minimum_height=results_text.setter('height'))
        results_text.height = max(len(text.split('\n')) * 30, 400)
        scroll_view.add_widget(results_text)
        
        # 按鈕區域
        btn_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        close_btn = ModernButton(
            text="關閉", 
            font_name=self.font_path, 
            background_color=(0.7, 0.3, 0.3, 1)
        )
        save_text_btn = ModernButton(
            text="保存結果文本", 
            font_name=self.font_path, 
            background_color=(0.3, 0.7, 0.3, 1)
        )
        
        btn_layout.add_widget(save_text_btn)
        btn_layout.add_widget(close_btn)
        
        layout.add_widget(scroll_view)
        layout.add_widget(btn_layout)
        
        popup = Popup(
            title="檢測結果", 
            content=layout, 
            size_hint=(0.9, 0.9),
            title_font=self.font_path, 
            title_align='center',
            title_size='18sp'
        )
        
        def save_text(instance):
            self.choose_text_save_location(text)
        
        close_btn.bind(on_press=popup.dismiss)
        save_text_btn.bind(on_press=save_text)
        popup.open()
    
    def choose_text_save_location(self, text_content):
        content = BoxLayout(orientation='vertical')
        
        # 預設檔名
        now = datetime.now()
        default_filename = f"pipe_detection_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 檔案選擇器
        file_chooser = FileChooserIconView(dirselect=True)
        
        # 檔名輸入
        filename_layout = BoxLayout(size_hint_y=None, height=40)
        filename_layout.add_widget(Label(text='檔案名稱:', size_hint_x=0.3,font_name='msjh.ttc'))
        filename_input = TextInput(text=default_filename, multiline=False, size_hint_x=0.7)
        filename_layout.add_widget(filename_input)
        
        # 按鈕
        btn_layout = BoxLayout(size_hint_y=None, height=50)
        save_btn = ModernButton(text='儲存',font_name=self.font_path)
        cancel_btn = ModernButton(text='取消',font_name=self.font_path)
        btn_layout.add_widget(save_btn)
        btn_layout.add_widget(cancel_btn)
        
        content.add_widget(Label(text='選擇儲存位置:',font_name='msjh.ttc'))
        content.add_widget(file_chooser)
        content.add_widget(filename_layout)
        content.add_widget(btn_layout)
        
        popup = Popup(title='儲存檢測結果文本', content=content, size_hint=(0.9, 0.9), title_font=self.font_path)
        
        def on_save(instance):
            if file_chooser.selection:
                directory = file_chooser.selection[0]
                if os.path.isdir(directory):
                    filepath = os.path.join(directory, filename_input.text)
                    try:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        self.show_popup("儲存成功", f"結果已儲存為 {filepath}")
                        popup.dismiss()
                    except Exception as e:
                        self.show_error(f"儲存失敗: {str(e)}")
                else:
                    self.show_error("請選擇有效的目錄")
            else:
                self.show_error("請選擇儲存位置")
        
        save_btn.bind(on_press=on_save)
        cancel_btn.bind(on_press=popup.dismiss)
        
        popup.open()
    def upload_to_web(self, data):
        try:
            files = {
                'grid_image': open(self.grid_image_path, 'rb'),
                'pipe_image': open(self.pipe_image_path, 'rb'),
                'output_image': ('output.jpg', cv2.imencode('.jpg', self.detection_result)[1].tobytes())
            }
            
            response = requests.post(
                'http://localhost:5000/api/upload',
                data=data,
                files=files
            )
            
            if response.status_code == 200:
                self.show_popup("上傳成功", "資料已儲存至資料庫")
            else:
                self.show_error(f"上傳失敗: {response.json()['message']}")
                
        except Exception as e:
            self.show_error(f"上傳錯誤: {str(e)}")
class SteelPipeDetectorApp(App):
    def build(self):
        # 設置窗口標題
        self.title = 'NFU 鋼管檢測系統 - 智慧型視覺運算實驗室'
        
        # 設置窗口大小
        Window.size = (1200, 900)
        
        # 設置窗口背景顏色
        Window.clearcolor = (0.1, 0.1, 0.12, 1)
        
        # 如果存在自定義圖標，設置應用圖標
        if os.path.exists('nfu_icon.png'):
            self.icon = 'nfu_icon.png'
        
        # 創建主應用程序
        return SteelPipeDetector()
    

if __name__ == '__main__':
    SteelPipeDetectorApp().run()
     