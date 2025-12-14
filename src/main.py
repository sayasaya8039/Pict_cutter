"""
Pict Cutter - 画像トリミング＆超解像アプリ
エントリーポイント
"""

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from main_window import MainWindow


def main():
    """メイン関数"""
    # High DPI対応
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # アプリケーション設定
    app.setApplicationName("Pict Cutter")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Pict Cutter")

    # フォント設定
    font = QFont("Yu Gothic UI", 10)
    app.setFont(font)

    # メインウィンドウを表示
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
