@echo off
echo 開始編譯應用程序...

echo 確保所有依賴已安裝...
pip install -r requirements.txt

echo 使用Nuitka進行編譯...
python -m nuitka --mingw64 --follow-imports --enable-plugin=numpy --enable-plugin=kivy --disable-console --windows-icon-from-ico=nfu_icon.ico --include-data-dir=./=./ --show-memory --show-progress --output-dir=build SteelPipeDetectorApp.py

echo 編譯完成，執行文件在 build 目錄下

pause