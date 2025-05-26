from flask import Flask, request, jsonify, render_template, send_from_directory,url_for,redirect # 匯入 Flask 主要類別與請求/響應工具
from flask_sqlalchemy import SQLAlchemy  # 匯入 SQLAlchemy 擴充套件，用於 ORM
from datetime import datetime  # 匯入 datetime，用於時間戳記
import os  # 匯入 os，用於檔案與目錄操作
import json  # 匯入 json，用於處理 JSON 資料
from werkzeug.utils import secure_filename  # 匯入 secure_filename，確保上傳檔名安全
from fpdf import FPDF  # 匯入 FPDF 類別，用於產生 PDF
from fpdf.enums import XPos, YPos  # 匯入 PDF 位置列舉
import platform  # 匯入 platform，用於檢查作業系統
import cv2, numpy as np, base64
from ultralytics import YOLO
from SteelPipeDetectorApp import detect_chessboard_calibration, detect_inner_circle_advanced
# 建立 Flask 應用實例
model = YOLO('best.pt')
app = Flask(__name__)  # __name__ 指定模組名稱，讓 Flask 知道資源路徑
# 設定 SQLite 資料庫路徑
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///steel_pipe.db'  # 設定 ORM 連接字串
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # 設定上傳目錄
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 關閉追蹤修改，提升效能
# 初始化 SQLAlchemy
db = SQLAlchemy(app)  # 將 Flask 應用與 SQLAlchemy 綁定

# 目錄初始化
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # 若上傳目錄不存在，則建立
# 根據作業系統選擇中文字體路徑
PDF_FONT_PATH = 'msjh.ttc' if platform.system() == 'Windows' else '/usr/share/fonts/msjh.ttc'  # Windows 與 Linux 字體路徑差異
# 設定 PDF 報告存放資料夾
PDF_SAVE_DIR = os.path.join('static', 'pdf_reports')  # 組合靜態目錄與 PDF 子目錄
os.makedirs(PDF_SAVE_DIR, exist_ok=True)  # 若報告資料夾不存在，則建立
#手機端上傳占存保留
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# 定義產生 PDF 的類別，繼承自 FPDF
# 定義產生 PDF 的類別，繼承自 FPDF
class SteelPipePDF(FPDF):
    PAGE_MARGIN = 10  # 頁邊距 10mm
    COLOR_SCHEME = {  # 定義顏色主題
        'primary': (79, 129, 189),  # 主色調
        'secondary': (44, 62, 80),  # 次色調
        'accent': (192, 80, 77),  # 強調色
        'bg_even': (245, 245, 245)  # 偶數列背景色
    }

    def __init__(self, record):  # 建構子接受一筆檢測記錄
        super().__init__('P', 'mm', 'A4')  # 呼叫父類別，設定直向、單位 mm、A4 大小
        self.record = record  # 儲存資料庫記錄物件
        self._init_layout()  # 初始化版面配置
        self.set_auto_page_break(True, margin=15)  # 自動分頁，底部保留 15mm

    def _init_layout(self):  # 內部方法：初始化版面相關設定
        # 設定 logo 相關參數
        self.logo = {
            'path': 'nfu_logo.png',  # logo 檔案路徑
            'x': self.PAGE_MARGIN,  # logo X 座標
            'y': 6,  # logo Y 座標
            'w': 18  # logo 寬度 mm
        }
        self.section_gap = 3  # 區塊之間的垂直間距 mm
        self._init_fonts()  # 初始化 PDF 字體

    def _init_fonts(self):  # 內部方法：載入中文字型
        self.add_font('MSJH','',''+PDF_FONT_PATH)  # 正常字型
        self.add_font('MSJH','B',''+PDF_FONT_PATH)  # 粗體
        self.add_font('MSJH','I',''+PDF_FONT_PATH)  # 斜體
        self.add_font('MSJH','BI',''+PDF_FONT_PATH)  # 粗斜體

    def header(self):  # 標頭樣式
        if os.path.exists(self.logo['path']):  # 若 logo 檔存在
            self.image(
                self.logo['path'],  # 圖片路徑
                x=self.logo['x'],  # X 座標
                y=self.logo['y'],  # Y 座標
                w=self.logo['w'],  # 設定寬度 mm
                keep_aspect_ratio=True  # 保持長寬比
            )

        # 設定標題位置，右移並垂直對齊 logo
        title_x = self.logo['x'] + 45 + self.logo['w'] + 10  # 設定標題 X 座標
        title_y = self.logo['y'] + 5  # 設定標題 Y 座標
        self.set_xy(title_x, title_y)  # 設定寫字起點

        self.set_font('MSJH', 'B', 18)  # 設定字體：微軟正黑體、粗體、18pt
        self.set_text_color(*self.COLOR_SCHEME['primary'])  # 設定字色為主色調
        self.cell(0, 10, '鋼管檢測分析報告', new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # 輸出標題文字

        self.set_draw_color(*self.COLOR_SCHEME['accent'])  # 設定繪圖顏色為強調色
        line_y = self.get_y()  # 取得當前游標垂直位置
        start_x = self.PAGE_MARGIN
        end_x = 210 - self.PAGE_MARGIN
        self.line(start_x, line_y +5 , end_x, line_y +5 )  # 畫橫線分隔
        self.ln(10)  # 換行 10mm

    def footer(self):  # 頁腳樣式
        self.set_y(-15)  # 距離底部 15mm 開始繪製
        self.set_font('MSJH','I',8)  # 設定字型：微軟正黑體、斜體、8pt
        self.set_text_color(150,150,150)  # 灰色字
        txt = f"報告編號：{self.record.id} │ 生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # 報告編號與時間
        self.cell(0,5,txt,0,0,'C')  # 置中輸出頁腳文字

    def build_content(self):  # 外部呼叫：組合內容
        self._add_summary_table()  # 加入概覽表格
        self._add_result_text()  # 加入結果說明文字
        self._add_images()  # 加入圖片區塊
    
    def _add_summary_table(self):  # 內部方法：建立概覽表格
        self.add_section('檢測概覽')  # 標題
        widths = [50, 50, 45, 45]  # 各欄寬度 mm
        keys = ['格子列數','格子行數','格邊長(cm)','像素密度(px/cm)']  # 表頭文字
        vals = [  # 取值
            str(self.record.grid_rows),
            str(self.record.grid_cols),
            f"{self.record.square_size:.2f}",
            f"{self.record.pixels_per_cm:.2f}"
        ]
        # 表頭
        self.set_font('MSJH','B',12)  # 粗體 12pt
        for i, key in enumerate(keys):  # 迴圈輸出各欄標題
            self.set_fill_color(*self.COLOR_SCHEME['accent'])  # 欄底色
            self.set_text_color(255,255,255)  # 白字
            self.cell(widths[i], 8, key, 1, 0, 'C', fill=True)  # 畫儲存格
        self.ln()  # 換行
        # 資料列
        self.set_font('MSJH','',11)  # 正常字 11pt
        self.set_text_color(0,0,0)  # 黑字
        for i, val in enumerate(vals):  # 輸出每個資料
            self.set_fill_color(*self.COLOR_SCHEME['bg_even'] if i%2==0 else (255,255,255))  # 交替底色
            self.cell(widths[i], 8, val, 1, 0, 'C', fill=True)  # 畫儲存格
        self.ln(15)  # 換行 10mm

    def _add_result_text(self):  # 內部方法：加入結果說明
        self.add_section('檢測結果說明')  # 標題
        self.set_font('MSJH','',11)  # 正常字 11pt
        self.set_text_color(0,0,0)  # 黑字
        self.multi_cell(0, 6, self.record.result_text or '無說明內容')  # 多行文字
        self.ln(50)  # 換行 5mm

    def _add_images(self):  # 內部方法：插入圖片
        # 圖片依序：grid_image, pipe_image, output_image
        imgs = [  # 標題與檔名對照
            ('棋盤格', self.record.grid_image),
            ('原圖', self.record.pipe_image),
            ('檢測輸出', self.record.output_image)
        ]
        max_img_height = 110  # 預估最大高度 mm
        margin_top = 3  # 頂部間隔 mmS
        y_offset = -12
        for title, fname in imgs:  # 迴圈處理每張圖
            if not fname:  # 若檔名為空，跳過
                continue
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)  # 組合完整檔案路徑
            if not os.path.exists(path):  # 若檔不存在，跳過
                continue

            # 空間不夠時換頁
            remaining = self.h - self.get_y() - 20  # 計算剩餘可用空間
            if remaining < (max_img_height + margin_top + 10):  # 若不足
                self.add_page()  # 新增頁面
                self.set_y(40 + y_offset)  # 重設 Y 座標，避免貼頂部

            self.add_section(title)  # 圖片區標題

            # 插入圖片置中、寬度 180mm
            img_x = (210 - 180) / 2  # 計算 X 座標置中
            self.set_y(self.get_y() + margin_top)  # 調整垂直位置
            self.image(path, x=img_x, w=180, keep_aspect_ratio=True)  # 插入圖片
            self.ln(10)  # 換行


    def add_section(self, title):  # 公共方法：畫區段標題
        self.set_font('MSJH','B',14)  # 粗體 14pt
        self.set_text_color(*self.COLOR_SCHEME['secondary'])  # 次色調
        self.cell(0,10,title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # 輸出區段標題
        self.set_draw_color(200,200,200)  # 灰色線條
        self.line(self.PAGE_MARGIN, self.get_y(), 200, self.get_y())  # 畫分隔線
        self.ln(self.section_gap)  # 換行指定間距

# 定義生成 PDF 的路由

@app.route('/export_pdf/<int:id>')
def export_pdf(id):  # 匯出 PDF 並下載
    try:
        record = DetectionRecord.query.get_or_404(id)  # 根據 ID 取得記錄，找不到回 404
        pdf = SteelPipePDF(record)  # 建立 PDF 實例
        pdf.add_page()  # 新增頁面
        pdf.build_content()  # 組合內容
        filename = f"pipe_report_{id}.pdf"  # 設定檔名
        save_path = os.path.join(PDF_SAVE_DIR, filename)  # 完整儲存路徑
        pdf.output(save_path)  # 輸出 PDF
        return send_from_directory(PDF_SAVE_DIR, filename, as_attachment=True)  # 下載
    except Exception as e:  # 若有例外
        app.logger.error(f'PDF生成失敗: {str(e)}')  # 紀錄錯誤
        return jsonify({"status": "error", "message": "報告生成失敗，請檢查數據格式"}), 500  # 回傳錯誤 JSON

# 定義資料模型
class DetectionRecord(db.Model):  # ORM 類別
    id = db.Column(db.Integer, primary_key=True)  # 主鍵，自增
    timestamp = db.Column(db.DateTime, default=datetime.now)  # 預設為當前時間
    grid_rows = db.Column(db.Integer)  # 格子列數
    grid_cols = db.Column(db.Integer)  # 格子行數
    square_size = db.Column(db.Float)  # 格邊長
    pixels_per_cm = db.Column(db.Float)  # 像素密度
    result_text = db.Column(db.Text)  # 結果說明文字
    grid_image = db.Column(db.String(200))  # Grid 參考圖檔名
    pipe_image = db.Column(db.String(200))  # 原圖檔名
    output_image = db.Column(db.String(200))  # 輸出圖檔名
#------------------------------Phone
# 理論值列表，用於尺寸校正
THEORY_OUTER = [20, 12, 10]
THEORY_INNER_MAP = {
    20: [10, 15, 12, 14],
    12: [6, 7],
    10: [6]
}

# 保存原始 bytes 檔案輔助函式
def save_bytes(data: bytes, prefix: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = secure_filename(f"{prefix}_{timestamp}.jpg")
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'wb') as f:
        f.write(data)
    return filename

# GET: 檢測表單
@app.route('/web_detect', methods=['GET'])
def web_detect_form():
    return render_template('detect_form.html')

# POST: 執行檢測並跳轉到 detail
@app.route('/web_detect', methods=['POST'])
def web_detect_process():
    # 讀取參數
    rows = int(request.form['grid_rows'])
    cols = int(request.form['grid_cols'])
    square_size = float(request.form['square_size'])

    # 讀取原始檔 bytes
    grid_file = request.files['grid_image']
    pipe_file = request.files['pipe_image']
    grid_bytes = grid_file.read()
    pipe_bytes = pipe_file.read()

    # 解碼成 OpenCV 影像
    grid_arr = np.frombuffer(grid_bytes, np.uint8)
    grid_img = cv2.imdecode(grid_arr, cv2.IMREAD_COLOR)
    pipe_arr = np.frombuffer(pipe_bytes, np.uint8)
    pipe_img = cv2.imdecode(pipe_arr, cv2.IMREAD_COLOR)

    # 存儲原始檔
    grid_filename = save_bytes(grid_bytes, 'grid')
    pipe_filename = save_bytes(pipe_bytes, 'pipe')

    # 校正
    pixels_per_cm, _ = detect_chessboard_calibration(
        grid_img, pattern_size=(cols, rows), square_size_cm=square_size
    )

    # YOLO 推論
    results = model.predict(pipe_img, imgsz=640, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 標註並計算尺寸
    annotated = pipe_img.copy()
    detection_text = ''
    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box)
        roi = pipe_img[y1:y2, x1:x2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r_outer = (x2 - x1) // 2

        # 偵測內圈
        inner = detect_inner_circle_advanced(
            roi, (roi.shape[1]//2, roi.shape[0]//2, r_outer)
        )
        ir = inner[2] if inner else int(r_outer * 0.7)

        # 換算成 mm
        outer_mm = (2 * r_outer / pixels_per_cm) * 10
        inner_mm = (2 * ir / pixels_per_cm) * 10

        # 理論值修正
        corrected_outer = min(THEORY_OUTER, key=lambda x: abs(x - outer_mm))
        inner_candidates = THEORY_INNER_MAP.get(corrected_outer, [])
        if inner_candidates:
            corrected_inner = min(inner_candidates, key=lambda x: abs(x - inner_mm))
        else:
            corrected_inner = inner_mm
        outer_mm = corrected_outer
        inner_mm = corrected_inner

        # 壁厚計算
        wall_mm = (outer_mm - inner_mm) / 2

        detection_text += (
            f"Pipe {i}: 外徑={outer_mm:.1f}mm 內徑={inner_mm:.1f}mm 壁厚={wall_mm:.1f}mm\n"
        )

        # 繪製標註
        cv2.circle(annotated, (cx, cy), r_outer, (255, 0, 0), 4)
        cv2.circle(annotated, (cx, cy), ir, (0, 0, 255), 4)
        cv2.putText(
            annotated, str(i), (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5
        )

    # 存標註結果圖 bytes
    _, buf = cv2.imencode('.jpg', annotated)
    output_bytes = buf.tobytes()
    output_filename = save_bytes(output_bytes, 'output')

    # 建立並存入資料庫記錄
    record = DetectionRecord(
        grid_rows=cols,
        grid_cols=rows,
        square_size=square_size,
        pixels_per_cm=pixels_per_cm,
        result_text=detection_text,
        grid_image=grid_filename,
        pipe_image=pipe_filename,
        output_image=output_filename
    )
    db.session.add(record)
    db.session.commit()

    # 轉向既有 detail 頁面
    return redirect(url_for('detail', id=record.id))
#-----------------
# 關鍵修復：同時保留新舊路由
@app.route('/api/static/uploads', methods=['POST'])  # 舊路由兼容
@app.route('/api/upload', methods=['POST'])          # 新標準路由
def upload_data():  # 接收前端上傳資料與檔案
    try:
        def save_file(file, prefix):  # 內部方法：存檔
            if file and file.filename != '':  # 若有檔案且名稱非空
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # 時間戳
                filename = f"{prefix}_{timestamp}.png"  # 命名格式
                filename = secure_filename(filename)  # 安全化檔名
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 完整路徑
                file.save(filepath)  # 儲存檔案
                return filename  # 回傳檔名
            return None  # 無檔案則回 None

        record = DetectionRecord(
            grid_rows=int(request.form['grid_rows']),  # 轉型後存入
            grid_cols=int(request.form['grid_cols']),
            square_size=float(request.form['square_size']),
            pixels_per_cm=float(request.form['pixels_per_cm']),
            result_text=request.form['result_text']  # 純文字
        )

        record.grid_image = save_file(request.files.get('grid_image'), 'grid')  # 儲存 grid 圖
        record.pipe_image = save_file(request.files.get('pipe_image'), 'pipe')  # 儲存 pipe 圖
        record.output_image = save_file(request.files.get('output_image'), 'output')  # 儲存 output 圖

        db.session.add(record)  # 新增至資料庫 session
        db.session.commit()  # 提交交易
        return jsonify({"status": "success", "id": record.id}), 200  # 回傳成功 JSON

    except Exception as e:  # 若有例外
        return jsonify({"status": "error", "message": str(e)}), 500  # 回傳錯誤 JSON

@app.route('/uploads/<filename>')  # 提供上傳檔案的靜態服務
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  # 回傳指定檔案

@app.route('/')  # 首頁路由，顯示所有記錄列表
def index():
    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()  # 依時間倒序查所有
    return render_template('index.html', records=records)  # 傳入列表至模板

@app.route('/detail/<int:id>')  # 詳細頁路由，顯示單筆記錄
def detail(id):
    record = DetectionRecord.query.get_or_404(id)  # 取得或 404
    return render_template('detail.html', record=record)  # 傳入單筆記錄
@app.route('/upload_form', methods=['GET'])
def upload_form():
    """提供一個 HTML 表單，讓手機、電腦瀏覽器都能上傳資料。"""
    return render_template('upload.html')
# 啟動應用
if __name__ == '__main__':
    with app.app_context():  # 建立應用上下文
        db.create_all()  # 建立資料表
    app.run(host='0.0.0.0', port=5000, debug=True)  # 啟動伺服器，開啟除錯模式
