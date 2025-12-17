"""
Icon Gene Form - アイコン生成＆トリミングアプリ
All-in-one version for PyInstaller
"""

import sys
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QSpinBox,
    QGroupBox, QComboBox, QCheckBox, QMessageBox,
    QStatusBar, QApplication, QDialog, QLineEdit,
    QTextEdit, QDialogButtonBox, QProgressDialog
)
from PySide6.QtCore import Qt, QRect, Signal, QPoint, QSize, QThread, QSettings
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon


# ============================================================
# 設定管理
# ============================================================

class SettingsManager:
    """設定管理クラス"""

    def __init__(self):
        self.settings = QSettings("IconGeneForm", "IconGeneForm")

    def get_api_key(self) -> str:
        return self.settings.value("api_key", "")

    def set_api_key(self, key: str):
        self.settings.setValue("api_key", key)

    def get_model(self) -> str:
        return self.settings.value("model", "nano-banana")

    def set_model(self, model: str):
        self.settings.setValue("model", model)


# ============================================================
# 設定ダイアログ
# ============================================================

class SettingsDialog(QDialog):
    """設定ダイアログ"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setMinimumWidth(400)
        self.settings_manager = SettingsManager()

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # APIキー設定
        api_group = QGroupBox("API設定")
        api_layout = QVBoxLayout(api_group)

        api_layout.addWidget(QLabel("Google AI Studio API キー:"))
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setEchoMode(QLineEdit.Password)
        self.txt_api_key.setPlaceholderText("APIキーを入力...")
        api_layout.addWidget(self.txt_api_key)

        # 表示/非表示ボタン
        self.btn_show_key = QPushButton("表示")
        self.btn_show_key.setCheckable(True)
        self.btn_show_key.clicked.connect(self._toggle_key_visibility)
        api_layout.addWidget(self.btn_show_key)

        layout.addWidget(api_group)

        # ボタン
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_key_visibility(self):
        if self.btn_show_key.isChecked():
            self.txt_api_key.setEchoMode(QLineEdit.Normal)
            self.btn_show_key.setText("隠す")
        else:
            self.txt_api_key.setEchoMode(QLineEdit.Password)
            self.btn_show_key.setText("表示")

    def _load_settings(self):
        self.txt_api_key.setText(self.settings_manager.get_api_key())

    def _save_and_accept(self):
        self.settings_manager.set_api_key(self.txt_api_key.text().strip())
        self.accept()


# ============================================================
# 画像生成ダイアログ
# ============================================================

class ImageGenerateDialog(QDialog):
    """画像生成ダイアログ"""

    image_generated = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI画像生成")
        self.setMinimumSize(500, 400)
        self.settings_manager = SettingsManager()
        self.generated_image = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # モデル選択
        model_group = QGroupBox("モデル選択")
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("モデル:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Gemini 2.5 Flash Image (標準)", "Gemini 3.0 Pro Image (高品質)"])
        model_layout.addWidget(self.combo_model)
        model_layout.addStretch()
        layout.addWidget(model_group)

        # サイズ選択
        size_group = QGroupBox("出力サイズ")
        size_layout = QHBoxLayout(size_group)
        size_layout.addWidget(QLabel("サイズ:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024"])
        self.combo_size.setCurrentIndex(2)  # デフォルト: 128x128
        size_layout.addWidget(self.combo_size)
        size_layout.addStretch()
        layout.addWidget(size_group)

        # プロンプト入力
        prompt_group = QGroupBox("プロンプト")
        prompt_layout = QVBoxLayout(prompt_group)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("生成したいアイコンの説明を入力...\n例: cute cat icon, simple flat design, white background")
        self.txt_prompt.setMaximumHeight(120)
        prompt_layout.addWidget(self.txt_prompt)
        layout.addWidget(prompt_group)

        # プレビュー
        preview_group = QGroupBox("プレビュー")
        preview_layout = QVBoxLayout(preview_group)
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(200, 200)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "background-color: #1E293B; border: 2px solid #334155; border-radius: 8px;"
        )
        self.lbl_preview.setText("生成された画像がここに表示されます")
        preview_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)
        layout.addWidget(preview_group)

        # ボタン
        btn_layout = QHBoxLayout()

        self.btn_generate = QPushButton("生成")
        self.btn_generate.clicked.connect(self._generate_image)
        btn_layout.addWidget(self.btn_generate)

        self.btn_use = QPushButton("この画像を使用")
        self.btn_use.clicked.connect(self._use_image)
        self.btn_use.setEnabled(False)
        btn_layout.addWidget(self.btn_use)

        btn_cancel = QPushButton("キャンセル")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _get_model_name(self) -> str:
        """選択されたモデル名を取得"""
        text = self.combo_model.currentText()
        if "3.0" in text:
            return "gemini-3-pro-image-preview"
        return "gemini-2.5-flash-image"

    def _get_size(self) -> int:
        """選択されたサイズを取得"""
        size_text = self.combo_size.currentText()
        return int(size_text.split("x")[0])

    def _generate_image(self):
        """画像を生成"""
        api_key = self.settings_manager.get_api_key()
        if not api_key:
            QMessageBox.warning(self, "エラー", "APIキーが設定されていません。\n設定画面でAPIキーを入力してください。")
            return

        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "エラー", "プロンプトを入力してください。")
            return

        model = self._get_model_name()
        size = self._get_size()

        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("生成中...")
        QApplication.processEvents()

        try:
            # Google AI Studio Imagen API呼び出し
            image = self._call_imagen_api(api_key, model, prompt, size)

            if image is not None:
                self.generated_image = image
                self._show_preview(image)
                self.btn_use.setEnabled(True)
            else:
                QMessageBox.warning(self, "エラー", "画像の生成に失敗しました。")

        except Exception as e:
            QMessageBox.warning(self, "エラー", f"生成エラー: {str(e)}")

        finally:
            self.btn_generate.setEnabled(True)
            self.btn_generate.setText("生成")

    def _call_imagen_api(self, api_key: str, model: str, prompt: str, size: int) -> Optional[np.ndarray]:
        """Google AI Studio Gemini APIを呼び出して画像を生成"""
        import base64

        try:
            # Gemini API エンドポイント（generateContent）
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

            # プロンプトにアイコンスタイルを追加
            full_prompt = f"Generate a {size}x{size} pixel icon image: {prompt}. Style: icon, 1:1 aspect ratio, centered, clean design, suitable for app icon."

            # リクエストデータ（Gemini形式）
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["image", "text"],
                    "responseMimeType": "text/plain"
                }
            }

            # リクエスト作成（APIキーはヘッダーで渡す）
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                },
                method="POST"
            )

            # API呼び出し
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))

                # レスポンスから画像データを取得（Gemini形式）
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "inlineData" in part:
                                img_data = base64.b64decode(part["inlineData"]["data"])
                                img_array = np.frombuffer(img_data, dtype=np.uint8)
                                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                                # 指定サイズにリサイズ
                                if image is not None:
                                    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
                                return image

                # エラーメッセージがあれば表示
                if "error" in result:
                    raise Exception(f"API Error: {result['error'].get('message', str(result['error']))}")

        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            raise Exception(f"API Error ({e.code}): {error_msg}")
        except urllib.error.URLError as e:
            raise Exception(f"接続エラー: {str(e.reason)}")
        except Exception as e:
            raise Exception(f"エラー: {str(e)}")

        return None

    def _show_preview(self, image: np.ndarray):
        """プレビュー表示"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_preview.setPixmap(pixmap)

    def _use_image(self):
        """生成した画像を使用"""
        if self.generated_image is not None:
            self.image_generated.emit(self.generated_image)
            self.accept()


# ============================================================
# 画像処理クラス
# ============================================================

class ImageProcessor:
    """画像処理クラス"""

    # 検出感度の設定値
    DETECTION_SETTINGS = {
        "ゆるい": {
            "canny_low": 30,      # 低い閾値 = より多くのエッジを検出
            "canny_high": 100,
            "dilate_iter": 3,     # 多めに膨張 = 広めに検出
            "erode_iter": 1,
            "margin": 15,         # 大きめのマージン
            "min_contour_area": 50,  # 小さいノイズも含める
        },
        "普通": {
            "canny_low": 50,
            "canny_high": 150,
            "dilate_iter": 2,
            "erode_iter": 1,
            "margin": 8,
            "min_contour_area": 100,
        },
        "きつい": {
            "canny_low": 80,      # 高い閾値 = 明確なエッジのみ
            "canny_high": 200,
            "dilate_iter": 1,     # 少なめに膨張 = タイトに検出
            "erode_iter": 2,      # 多めに収縮 = ノイズ除去
            "margin": 2,          # 小さめのマージン
            "min_contour_area": 200,  # 大きい輪郭のみ
        },
    }

    def __init__(self):
        self.upscaler = None
        self.upscaler_available = False

    def detect_content_bounds(self, image: np.ndarray, mode: str = "普通") -> Tuple[int, int, int, int]:
        """
        AIによる余白検知 - コンテンツの境界を検出

        Args:
            image: 入力画像
            mode: 検出モード ("ゆるい", "普通", "きつい")
        """
        settings = self.DETECTION_SETTINGS.get(mode, self.DETECTION_SETTINGS["普通"])

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # エッジ検出（感度に応じたパラメータ）
        edges = cv2.Canny(gray, settings["canny_low"], settings["canny_high"])

        # 四隅のウォーターマーク/ロゴを除外（画像サイズの12%をマスク）
        img_h, img_w = edges.shape[:2]
        corner_w = int(img_w * 0.12)
        corner_h = int(img_h * 0.12)

        # 左上
        edges[0:corner_h, 0:corner_w] = 0
        # 右上
        edges[0:corner_h, img_w-corner_w:img_w] = 0
        # 左下
        edges[img_h-corner_h:img_h, 0:corner_w] = 0
        # 右下
        edges[img_h-corner_h:img_h, img_w-corner_w:img_w] = 0

        # モルフォロジー処理
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=settings["dilate_iter"])
        edges = cv2.erode(edges, kernel, iterations=settings["erode_iter"])

        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            h, w = image.shape[:2]
            return 0, 0, w, h

        # 最小面積でフィルタリング
        min_area = settings["min_contour_area"]
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not filtered_contours:
            filtered_contours = contours  # フィルタ後に空なら元の輪郭を使用

        all_points = np.vstack(filtered_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # マージンを追加
        margin = settings["margin"]
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return x, y, w, h

    def get_square_crop_region(self, image: np.ndarray, mode: str = "普通") -> Tuple[int, int, int, int]:
        """
        1:1の正方形クロップ領域を取得（余白検知ベース）

        Args:
            image: 入力画像
            mode: 検出モード ("ゆるい", "普通", "きつい")
        """
        x, y, w, h = self.detect_content_bounds(image, mode)
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
            elif format.upper() == 'ICO':
                # ICO形式はPillowで保存
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                # ICOは複数サイズを含めることが可能、ここでは単一サイズで保存
                pil_image.save(path, format='ICO')
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
        self.setWindowTitle("Icon Gene Form - アイコン生成＆トリミング")
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
        self.current_image_name: str = "output"  # 元画像のファイル名（拡張子なし）
        self.last_save_dir: str = ""  # 最後に保存したフォルダ

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

        # 2行目のボタン
        btn_layout2 = QHBoxLayout()

        self.btn_generate = QPushButton("AI画像生成")
        self.btn_generate.clicked.connect(self._open_generate_dialog)
        btn_layout2.addWidget(self.btn_generate)

        self.btn_settings = QPushButton("設定")
        self.btn_settings.clicked.connect(self._open_settings)
        btn_layout2.addWidget(self.btn_settings)

        left_panel.addLayout(btn_layout2)

        main_layout.addLayout(left_panel, 2)

        # 右側: 設定＆プレビュー
        right_panel = QVBoxLayout()

        # 自動検出設定
        detect_group = QGroupBox("自動検出設定")
        detect_layout = QVBoxLayout(detect_group)

        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("検出感度:"))
        self.combo_sensitivity = QComboBox()
        self.combo_sensitivity.addItems(["ゆるい", "普通", "きつい"])
        self.combo_sensitivity.setCurrentIndex(1)  # デフォルト: 普通
        sensitivity_layout.addWidget(self.combo_sensitivity)
        sensitivity_layout.addStretch()
        detect_layout.addLayout(sensitivity_layout)

        # 感度の説明
        self.lbl_sensitivity_desc = QLabel("余白を適度に残して検出")
        self.lbl_sensitivity_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        detect_layout.addWidget(self.lbl_sensitivity_desc)
        self.combo_sensitivity.currentTextChanged.connect(self._on_sensitivity_changed)

        right_panel.addWidget(detect_group)

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
        self.combo_output_size = QComboBox()
        self.combo_output_size.addItems(["32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024"])
        self.combo_output_size.setCurrentIndex(2)  # デフォルト: 128x128
        size_layout.addWidget(self.combo_output_size)
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
        self.combo_format.addItems(["PNG", "JPEG", "ICO"])
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
            self.current_image_name = Path(path).stem  # 拡張子なしのファイル名を保存
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

    def _open_generate_dialog(self):
        """AI画像生成ダイアログを開く"""
        dialog = ImageGenerateDialog(self)
        dialog.image_generated.connect(self._on_image_generated)
        dialog.exec()

    def _on_image_generated(self, image: np.ndarray):
        """生成された画像を受け取る"""
        self.current_image = image
        self.current_image_name = "generated_icon"
        self._display_image(image)
        self.btn_auto_detect.setEnabled(True)
        self.btn_reset.setEnabled(True)

        # 画像全体を選択範囲として設定
        h, w = image.shape[:2]
        self.image_label.set_selection(0, 0, w, h)

        self.statusbar.showMessage(f"AI画像生成完了: {w}x{h}")

    def _open_settings(self):
        """設定ダイアログを開く"""
        dialog = SettingsDialog(self)
        dialog.exec()

    def _auto_detect(self):
        if self.current_image is None:
            return
        mode = self.combo_sensitivity.currentText()
        x, y, w, h = self.processor.get_square_crop_region(self.current_image, mode)
        self.image_label.set_selection(x, y, w, h)
        self.statusbar.showMessage(f"自動検出 [{mode}]: 選択範囲 ({x}, {y}) {w}x{h}")

    def _on_sensitivity_changed(self, mode: str):
        """検出感度変更時の処理"""
        descriptions = {
            "ゆるい": "余白を多めに残して広く検出",
            "普通": "余白を適度に残して検出",
            "きつい": "余白を最小限にしてタイトに検出",
        }
        self.lbl_sensitivity_desc.setText(descriptions.get(mode, ""))

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
        target_size = int(self.combo_output_size.currentText().split("x")[0])

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
        ext = ".png" if format_str == "PNG" else ".ico" if format_str == "ICO" else ".jpg"

        # 元画像のファイル名をベースにしたデフォルト名（前回の保存先フォルダを使用）
        default_name = f"{self.current_image_name}_cropped{ext}"
        if self.last_save_dir:
            default_name = str(Path(self.last_save_dir) / default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", default_name, f"{format_str}ファイル (*{ext})"
        )

        if path:
            quality = self.slider_quality.value()
            success = self.processor.save_image(self.upscaled_image, path, format_str, quality)

            if success:
                self.last_save_dir = str(Path(path).parent)
                self.statusbar.showMessage(f"保存完了: {path}")
                QMessageBox.information(self, "完了", f"画像を保存しました:\n{path}")
            else:
                QMessageBox.warning(self, "エラー", "保存に失敗しました")


# ============================================================
# エントリーポイント
# ============================================================

def create_app_icon() -> QIcon:
    """アプリアイコンを生成"""
    import math
    size = 256
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)

    # 背景
    bg_color = QColor(15, 23, 42)
    accent_color = QColor(56, 189, 248)
    highlight_color = QColor(125, 211, 252)

    # 角丸背景
    painter.setPen(QPen(accent_color, 4))
    painter.setBrush(bg_color)
    painter.drawRoundedRect(20, 20, size-40, size-40, 40, 40)

    # フレーム
    painter.setPen(QPen(highlight_color, 3))
    painter.setBrush(Qt.transparent)
    painter.drawRoundedRect(50, 50, size-100, size-100, 20, 20)

    # 山形
    from PySide6.QtGui import QPolygon
    from PySide6.QtCore import QPoint as QP
    points = [QP(70, 186), QP(110, 120), QP(140, 150), QP(170, 100), QP(186, 186)]
    painter.setPen(Qt.NoPen)
    painter.setBrush(accent_color)
    painter.drawPolygon(QPolygon(points))

    # 太陽
    painter.setBrush(highlight_color)
    painter.drawEllipse(156, 60, 40, 40)

    # スパーク
    spark_x, spark_y = 196, 60
    painter.setPen(QPen(highlight_color, 2))
    for angle in range(0, 360, 45):
        x1 = int(spark_x + 8 * math.cos(math.radians(angle)))
        y1 = int(spark_y + 8 * math.sin(math.radians(angle)))
        x2 = int(spark_x + 15 * math.cos(math.radians(angle)))
        y2 = int(spark_y + 15 * math.sin(math.radians(angle)))
        painter.drawLine(x1, y1, x2, y2)

    painter.end()
    return QIcon(QPixmap.fromImage(img))


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Icon Gene Form")
    app.setApplicationVersion("1.0.0")

    # アプリアイコンを設定
    app.setWindowIcon(create_app_icon())

    font = QFont("Yu Gothic UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
