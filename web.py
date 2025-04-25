from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import json
from werkzeug.utils import secure_filename
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import platform

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///steel_pipe.db'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 目錄初始化
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
PDF_FONT_PATH = 'msjh.ttc' if platform.system() == 'Windows' else '/usr/share/fonts/msjh.ttc'
PDF_SAVE_DIR = os.path.join('static', 'pdf_reports')
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

class SteelPipePDF(FPDF):
    PAGE_MARGIN = 10
    COLOR_SCHEME = {
        'primary': (79, 129, 189),
        'secondary': (44, 62, 80),
        'accent': (192, 80, 77),
        'bg_even': (245, 245, 245)
    }

    def __init__(self, record):
        super().__init__('P', 'mm', 'A4')
        self.record = record
        self._init_layout()
        self.set_auto_page_break(True, margin=15)

    def _init_layout(self):
        # 將 logo 寬度設定為 25mm，Y 軸下移到 15mm
        self.logo = {
            'path': 'nfu_logo.png',
            'x': self.PAGE_MARGIN,
            'y': 15,
            'w': 18
        }
        self.section_gap = 6
        self._init_fonts()

    def _init_fonts(self):
        self.add_font('MSJH','',''+PDF_FONT_PATH)
        self.add_font('MSJH','B',''+PDF_FONT_PATH)
        self.add_font('MSJH','I',''+PDF_FONT_PATH)
        self.add_font('MSJH','BI',''+PDF_FONT_PATH)

    def header(self):
        if os.path.exists(self.logo['path']):
            self.image(
                self.logo['path'],
                x=self.logo['x'],
                y=self.logo['y'],
                w=self.logo['w'],
                keep_aspect_ratio=True
            )

        # 調整標題位置靠右並垂直對齊 logo
        title_x = self.logo['x'] + self.logo['w'] + 10
        title_y = self.logo['y'] + 10
        self.set_xy(title_x, title_y)

        self.set_font('MSJH', 'B', 18)
        self.set_text_color(*self.COLOR_SCHEME['primary'])
        self.cell(0, 10, '鋼管檢測分析報告', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_draw_color(*self.COLOR_SCHEME['accent'])
        self.line(self.PAGE_MARGIN, 35, 210 - self.PAGE_MARGIN, 35)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('MSJH','I',8)
        self.set_text_color(150,150,150)
        txt = f"報告編號：{self.record.id} │ 生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.cell(0,5,txt,0,0,'C')

    def build_content(self):
        self._add_summary_table()
        self._add_result_text()
        self._add_images()
    
    def _add_summary_table(self):
        self.add_section('檢測概覽')
        widths = [50, 50, 45, 45]
        keys = ['格子列數','格子行數','格邊長(cm)','像素密度(px/cm)']
        vals = [
            str(self.record.grid_rows),
            str(self.record.grid_cols),
            f"{self.record.square_size:.2f}",
            f"{self.record.pixels_per_cm:.2f}"
        ]
        # 表頭
        self.set_font('MSJH','B',12)
        for i, key in enumerate(keys):
            self.set_fill_color(*self.COLOR_SCHEME['accent'])
            self.set_text_color(255,255,255)
            self.cell(widths[i], 8, key, 1, 0, 'C', fill=True)
        self.ln()
        # 資料列
        self.set_font('MSJH','',11)
        self.set_text_color(0,0,0)
        for i, val in enumerate(vals):
            self.set_fill_color(*self.COLOR_SCHEME['bg_even'] if i%2==0 else (255,255,255))
            self.cell(widths[i], 8, val, 1, 0, 'C', fill=True)
        self.ln(10)

    def _add_result_text(self):
        self.add_section('檢測結果說明')
        self.set_font('MSJH','',11)
        self.set_text_color(0,0,0)
        self.multi_cell(0, 6, self.record.result_text or '無說明內容')
        self.ln(5)

    def _add_images(self):
        # 圖片依序：grid_image, pipe_image, output_image
        imgs = [
            ('格子參考圖', self.record.grid_image),
            ('原圖', self.record.pipe_image),
            ('檢測輸出', self.record.output_image)
        ]
        max_img_height = 110  # 預估圖片最大高度（mm）
        margin_top = 10
        for title, fname in imgs:
            if not fname:
                continue
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if not os.path.exists(path):
                continue

            # 空間不夠時換頁
            remaining = self.h - self.get_y() - 20  # 留footer空間
            if remaining < (max_img_height + margin_top + 10):
                self.add_page()
                self.set_y(40)  # 避免貼頂部

            self.add_section(title)

            # 插入圖片置中、寬度 180mm
            img_x = (210 - 180) / 2
            self.set_y(self.get_y() + margin_top)
            self.image(path, x=img_x, w=180, keep_aspect_ratio=True)
            self.ln(10)


    def add_section(self, title):
        self.set_font('MSJH','B',14)
        self.set_text_color(*self.COLOR_SCHEME['secondary'])
        self.cell(0,10,title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200,200,200)
        self.line(self.PAGE_MARGIN, self.get_y(), 200, self.get_y())
        self.ln(self.section_gap)

@app.route('/export_pdf/<int:id>')
def export_pdf(id):
    try:
        record = DetectionRecord.query.get_or_404(id)
        pdf = SteelPipePDF(record)
        pdf.add_page()
        pdf.build_content()
        filename = f"pipe_report_{id}.pdf"
        save_path = os.path.join(PDF_SAVE_DIR, filename)
        pdf.output(save_path)
        return send_from_directory(PDF_SAVE_DIR, filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f'PDF生成失敗: {str(e)}')
        return jsonify({"status": "error", "message": "報告生成失敗，請檢查數據格式"}), 500

class DetectionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    grid_rows = db.Column(db.Integer)
    grid_cols = db.Column(db.Integer)
    square_size = db.Column(db.Float)
    pixels_per_cm = db.Column(db.Float)
    result_text = db.Column(db.Text)
    grid_image = db.Column(db.String(200))
    pipe_image = db.Column(db.String(200))
    output_image = db.Column(db.String(200))

# 關鍵修復：同時保留新舊路由
@app.route('/api/static/uploads', methods=['POST'])  # 舊路由兼容
@app.route('/api/upload', methods=['POST'])          # 新標準路由
def upload_data():
    try:
        def save_file(file, prefix):
            if file and file.filename != '':
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"{prefix}_{timestamp}.png"
                filename = secure_filename(filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                return filename
            return None

        record = DetectionRecord(
            grid_rows=int(request.form['grid_rows']),
            grid_cols=int(request.form['grid_cols']),
            square_size=float(request.form['square_size']),
            pixels_per_cm=float(request.form['pixels_per_cm']),
            result_text=request.form['result_text']
        )

        record.grid_image = save_file(request.files.get('grid_image'), 'grid')
        record.pipe_image = save_file(request.files.get('pipe_image'), 'pipe')
        record.output_image = save_file(request.files.get('output_image'), 'output')

        db.session.add(record)
        db.session.commit()
        return jsonify({"status": "success", "id": record.id}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    records = DetectionRecord.query.order_by(DetectionRecord.timestamp.desc()).all()
    return render_template('index.html', records=records)

@app.route('/detail/<int:id>')
def detail(id):
    record = DetectionRecord.query.get_or_404(id)
    return render_template('detail.html', record=record)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)

