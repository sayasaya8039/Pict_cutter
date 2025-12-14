"""
Pict Cutter - 画像トリミング＆超解像アプリ
All-in-one version for PyInstaller
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QSpinBox,
    QGroupBox, QComboBox, QCheckBox, QMessageBox,
    QStatusBar, QApplication
)
from PySide6.QtCore import Qt, QRect, Signal, QPoint, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont


# ============================================================
# 画像処理クラス
# ============================================================

class ImageProcessor:
    """画像処理クラス"""

    def __init__(self):
        self.upscaler = None
        self.upscaler_available = False

    def detect_content_bounds(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
        """AIによる余白検知 - コンテンツの境界を検出"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            h, w = image.shape[:2]
            return 0, 0, w, h

        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return x, y, w, h

    def get_square_crop_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """1:1の正方形クロップ領域を取得（余白検知ベース）"""
        x, y, w, h = self.detect_content_bounds(image)
        size = max(w, h)

        center_x = x + w // 2
        center_y = y + h // 2

        new_x = max(0, center_x - size // 2)
        new_y = max(0, center_y - size // 2)

        img_h, img_w = image.shape[:2]
        if new_x + size > img_w:
            new_x = max(0, img_w - size)
        if new_y + size > img_h:
            new_y = max(0, img_h - size)

        final_size = min(size, img_w - new_x, img_h - new_y)
        return new_x, new_y, final_size, final_size

    def crop_image(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """画像をトリミング"""
        return image[y:y+h, x:x+w].copy()

    def upscale_image(self, image: np.ndarray, target_size: int = 128) -> np.ndarray:
        """高品質な画像拡大"""
        return self._high_quality_resize(image, target_size)

    def _high_quality_resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """高品質な画像リサイズ"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resized = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

    def save_image(self, image: np.ndarray, path: str, format: str = 'PNG', quality: int = 95) -> bool:
        """画像を保存"""
        try:
            if format.upper() == 'PNG':
                cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
        except Exception as e:
            print(f"保存エラー: {e}")
            return False

    def load_image(self, path: str) -> Optional[np.ndarray]:
        """画像を読み込み"""
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"読み込みエラー: {e}")
            return None


# ============================================================
# カスタム画像ラベル
# ============================================================

class ImageLabel(QLabel):
    """画像表示用のカスタムラベル（選択範囲表示機能付き）"""

    selection_changed = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1E293B; border: 2px solid #38BDF8; border-radius: 8px;")

        self._pixmap: Optional[QPixmap] = None
        self._selection: Optional[QRect] = None
        self._dragging = False
        self._drag_start = QPoint()
        self._image_rect = QRect()

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._selection = None
        self.update()

    def set_selection(self, x: int, y: int, w: int, h: int):
        self._selection = QRect(x, y, w, h)
        self.update()
        self.selection_changed.emit(x, y, w, h)

    def get_selection(self) -> Optional[Tuple[int, int, int, int]]:
        if self._selection:
            return (self._selection.x(), self._selection.y(),
                    self._selection.width(), self._selection.height())
        return None

    def clear_selection(self):
        self._selection = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1E293B"))

        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            self._image_rect = QRect(x, y, scaled.width(), scaled.height())

            painter.drawPixmap(self._image_rect, scaled)

            if self._selection and self._pixmap:
                scale_x = scaled.width() / self._pixmap.width()
                scale_y = scaled.height() / self._pixmap.height()

                sel_x = int(x + self._selection.x() * scale_x)
                sel_y = int(y + self._selection.y() * scale_y)
                sel_w = int(self._selection.width() * scale_x)
                sel_h = int(self._selection.height() * scale_y)

                overlay = QColor(0, 0, 0, 128)
                painter.fillRect(x, y, scaled.width(), sel_y - y, overlay)
                painter.fillRect(x, sel_y + sel_h, scaled.width(), y + scaled.height() - sel_y - sel_h, overlay)
                painter.fillRect(x, sel_y, sel_x - x, sel_h, overlay)
                painter.fillRect(sel_x + sel_w, sel_y, x + scaled.width() - sel_x - sel_w, sel_h, overlay)

                pen = QPen(QColor("#7DD3FC"), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(sel_x, sel_y, sel_w, sel_h)
        else:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignCenter, "画像をドラッグ＆ドロップ\nまたは「開く」ボタンで読み込み")

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap:
            self._dragging = True
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self._dragging and self._pixmap:
            pos = event.pos()
            start = self._drag_start

            if self._image_rect.width() > 0:
                scale_x = self._pixmap.width() / self._image_rect.width()
                scale_y = self._pixmap.height() / self._image_rect.height()

                img_x1 = int((min(start.x(), pos.x()) - self._image_rect.x()) * scale_x)
                img_y1 = int((min(start.y(), pos.y()) - self._image_rect.y()) * scale_y)
                img_x2 = int((max(start.x(), pos.x()) - self._image_rect.x()) * scale_x)
                img_y2 = int((max(start.y(), pos.y()) - self._image_rect.y()) * scale_y)

                size = max(abs(img_x2 - img_x1), abs(img_y2 - img_y1))

                img_x1 = max(0, min(img_x1, self._pixmap.width() - size))
                img_y1 = max(0, min(img_y1, self._pixmap.height() - size))

                self._selection = QRect(img_x1, img_y1, size, size)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            if self._selection:
                self.selection_changed.emit(
                    self._selection.x(), self._selection.y(),
                    self._selection.width(), self._selection.height()
                )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            parent = self.parent()
            while parent:
                if hasattr(parent, 'load_image'):
                    parent.load_image(path)
                    break
                parent = parent.parent()


# ============================================================
# メインウィンドウ
# ============================================================

class MainWindow(QMainWindow):
    """メインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pict Cutter - 画像トリミング＆超解像")
        self.setMinimumSize(1000, 700)

        self.setStyleSheet("""
            QMainWindow { background-color: #0F172A; }
            QWidget { color: #E0F2FE; font-family: "Yu Gothic UI", "Meiryo UI", sans-serif; }
            QPushButton {
                background-color: #38BDF8; color: #0F172A; border: none;
                border-radius: 6px; padding: 10px 20px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #7DD3FC; }
            QPushButton:pressed { background-color: #0EA5E9; }
            QPushButton:disabled { background-color: #475569; color: #94A3B8; }
            QGroupBox {
                background-color: #1E293B; border: 1px solid #334155;
                border-radius: 8px; margin-top: 12px; padding: 15px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; color: #7DD3FC; }
            QComboBox {
                background-color: #334155; border: 1px solid #475569;
                border-radius: 4px; padding: 8px; min-width: 100px;
            }
            QComboBox::drop-down { border: none; }
            QSpinBox {
                background-color: #334155; border: 1px solid #475569;
                border-radius: 4px; padding: 8px;
            }
            QSlider::groove:horizontal { background-color: #334155; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal {
                background-color: #38BDF8; width: 18px; height: 18px; margin: -5px 0; border-radius: 9px;
            }
            QSlider::sub-page:horizontal { background-color: #7DD3FC; border-radius: 4px; }
            QCheckBox { spacing: 8px; }
            QCheckBox::indicator { width: 20px; height: 20px; border-radius: 4px; border: 2px solid #475569; }
            QCheckBox::indicator:checked { background-color: #38BDF8; border-color: #38BDF8; }
            QStatusBar { background-color: #1E293B; color: #94A3B8; }
            QLabel { color: #E0F2FE; }
        """)

        self.processor = ImageProcessor()
        self.current_image: Optional[np.ndarray] = None
        self.cropped_image: Optional[np.ndarray] = None
        self.upscaled_image: Optional[np.ndarray] = None

        self._setup_ui()
        self._setup_statusbar()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 左側: 元画像表示
        left_panel = QVBoxLayout()

        self.image_label = ImageLabel()
        self.image_label.setAcceptDrops(True)
        self.image_label.selection_changed.connect(self._on_selection_changed)
        left_panel.addWidget(self.image_label, 1)

        btn_layout = QHBoxLayout()

        self.btn_open = QPushButton("画像を開く")
        self.btn_open.clicked.connect(self._open_image)
        btn_layout.addWidget(self.btn_open)

        self.btn_auto_detect = QPushButton("自動検出")
        self.btn_auto_detect.clicked.connect(self._auto_detect)
        self.btn_auto_detect.setEnabled(False)
        btn_layout.addWidget(self.btn_auto_detect)

        self.btn_reset = QPushButton("リセット")
        self.btn_reset.clicked.connect(self._reset_selection)
        self.btn_reset.setEnabled(False)
        btn_layout.addWidget(self.btn_reset)

        left_panel.addLayout(btn_layout)
        main_layout.addLayout(left_panel, 2)

        # 右側: 設定＆プレビュー
        right_panel = QVBoxLayout()

        # 選択範囲情報
        selection_group = QGroupBox("選択範囲")
        selection_layout = QVBoxLayout(selection_group)
        self.lbl_selection_info = QLabel("選択なし")
        selection_layout.addWidget(self.lbl_selection_info)
        right_panel.addWidget(selection_group)

        # 処理設定
        process_group = QGroupBox("処理設定")
        process_layout = QVBoxLayout(process_group)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("出力サイズ:"))
        self.spin_output_size = QSpinBox()
        self.spin_output_size.setRange(32, 512)
        self.spin_output_size.setValue(128)
        self.spin_output_size.setSuffix(" px")
        size_layout.addWidget(self.spin_output_size)
        size_layout.addStretch()
        process_layout.addLayout(size_layout)

        self.chk_upscale = QCheckBox("高品質リサイズを使用")
        self.chk_upscale.setChecked(True)
        process_layout.addWidget(self.chk_upscale)

        self.btn_process = QPushButton("トリミング＆拡大")
        self.btn_process.clicked.connect(self._process_image)
        self.btn_process.setEnabled(False)
        process_layout.addWidget(self.btn_process)

        right_panel.addWidget(process_group)

        # プレビュー
        preview_group = QGroupBox("プレビュー (128x128)")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(160, 160)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #0F172A; border: 2px solid #334155; border-radius: 8px;"
        )
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        right_panel.addWidget(preview_group)

        # 保存
        save_group = QGroupBox("保存")
        save_layout = QVBoxLayout(save_group)

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("形式:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG", "JPEG"])
        format_layout.addWidget(self.combo_format)
        format_layout.addStretch()
        save_layout.addLayout(format_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("品質:"))
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(1, 100)
        self.slider_quality.setValue(95)
        quality_layout.addWidget(self.slider_quality)
        self.lbl_quality = QLabel("95")
        quality_layout.addWidget(self.lbl_quality)
        self.slider_quality.valueChanged.connect(lambda v: self.lbl_quality.setText(str(v)))
        save_layout.addLayout(quality_layout)

        self.btn_save = QPushButton("保存")
        self.btn_save.clicked.connect(self._save_image)
        self.btn_save.setEnabled(False)
        save_layout.addWidget(self.btn_save)

        right_panel.addWidget(save_group)
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("画像を読み込んでください")

    def load_image(self, path: str):
        image = self.processor.load_image(path)
        if image is not None:
            self.current_image = image
            self._display_image(image)
            self.btn_auto_detect.setEnabled(True)
            self.btn_reset.setEnabled(True)
            self.statusbar.showMessage(f"読み込み完了: {Path(path).name} ({image.shape[1]}x{image.shape[0]})")
        else:
            QMessageBox.warning(self, "エラー", "画像の読み込みに失敗しました")

    def _display_image(self, image: np.ndarray):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.set_image(pixmap)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "画像を選択", "",
            "画像ファイル (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;すべてのファイル (*.*)"
        )
        if path:
            self.load_image(path)

    def _auto_detect(self):
        if self.current_image is None:
            return
        x, y, w, h = self.processor.get_square_crop_region(self.current_image)
        self.image_label.set_selection(x, y, w, h)
        self.statusbar.showMessage(f"自動検出: 選択範囲 ({x}, {y}) {w}x{h}")

    def _reset_selection(self):
        self.image_label.clear_selection()
        self.cropped_image = None
        self.upscaled_image = None
        self.preview_label.clear()
        self.btn_process.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.lbl_selection_info.setText("選択なし")
        self.statusbar.showMessage("選択範囲をリセットしました")

    def _on_selection_changed(self, x: int, y: int, w: int, h: int):
        self.lbl_selection_info.setText(f"位置: ({x}, {y})  サイズ: {w}x{h}")
        self.btn_process.setEnabled(True)

    def _process_image(self):
        if self.current_image is None:
            return

        selection = self.image_label.get_selection()
        if not selection:
            QMessageBox.warning(self, "警告", "選択範囲を指定してください")
            return

        x, y, w, h = selection
        target_size = self.spin_output_size.value()

        self.statusbar.showMessage("処理中...")
        QApplication.processEvents()

        self.cropped_image = self.processor.crop_image(self.current_image, x, y, w, h)
        self.upscaled_image = self.processor.upscale_image(self.cropped_image, target_size)

        self._show_preview(self.upscaled_image)
        self.btn_save.setEnabled(True)
        self.statusbar.showMessage(f"処理完了: {target_size}x{target_size}")

    def _show_preview(self, image: np.ndarray):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)

    def _save_image(self):
        if self.upscaled_image is None:
            return

        format_str = self.combo_format.currentText()
        ext = ".png" if format_str == "PNG" else ".jpg"

        path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", f"output{ext}", f"{format_str}ファイル (*{ext})"
        )

        if path:
            quality = self.slider_quality.value()
            success = self.processor.save_image(self.upscaled_image, path, format_str, quality)

            if success:
                self.statusbar.showMessage(f"保存完了: {path}")
                QMessageBox.information(self, "完了", f"画像を保存しました:\n{path}")
            else:
                QMessageBox.warning(self, "エラー", "保存に失敗しました")


# ============================================================
# エントリーポイント
# ============================================================

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Pict Cutter")
    app.setApplicationVersion("1.0.0")

    font = QFont("Yu Gothic UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
