"""
画像処理モジュール - 余白検知、トリミング、超解像
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import os


class ImageProcessor:
    """画像処理クラス"""

    def __init__(self):
        self.upscaler = None
        self._init_upscaler()

    def _init_upscaler(self):
        """Real-ESRGANの初期化"""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch

            # モデルパスの設定
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'RealESRGAN_x4plus.pth')

            # モデルが存在しない場合はダウンロード用のフラグを設定
            if not os.path.exists(model_path):
                self.upscaler = None
                self.upscaler_available = False
                return

            # RRDBNetモデルの設定
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            # デバイスの選択
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.upscaler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device
            )
            self.upscaler_available = True

        except Exception as e:
            print(f"Real-ESRGAN初期化エラー: {e}")
            self.upscaler = None
            self.upscaler_available = False

    def detect_content_bounds(self, image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
        """
        AIによる余白検知 - コンテンツの境界を検出

        Args:
            image: 入力画像 (BGR形式)
            threshold: 余白判定のしきい値

        Returns:
            (x, y, width, height) のタプル
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)

        # モルフォロジー処理でノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 輪郭が見つからない場合は画像全体を返す
            h, w = image.shape[:2]
            return 0, 0, w, h

        # 全輪郭を含む最小矩形を計算
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # 少しマージンを追加
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return x, y, w, h

    def get_square_crop_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        1:1の正方形クロップ領域を取得（余白検知ベース）

        Args:
            image: 入力画像

        Returns:
            (x, y, width, height) のタプル（正方形）
        """
        x, y, w, h = self.detect_content_bounds(image)

        # 正方形にするため、大きい方に合わせる
        size = max(w, h)

        # 中心を維持しながら正方形にする
        center_x = x + w // 2
        center_y = y + h // 2

        new_x = max(0, center_x - size // 2)
        new_y = max(0, center_y - size // 2)

        # 画像の境界を超えないように調整
        img_h, img_w = image.shape[:2]
        if new_x + size > img_w:
            new_x = max(0, img_w - size)
        if new_y + size > img_h:
            new_y = max(0, img_h - size)

        # 最終的なサイズを調整
        final_size = min(size, img_w - new_x, img_h - new_y)

        return new_x, new_y, final_size, final_size

    def crop_image(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        画像をトリミング

        Args:
            image: 入力画像
            x, y: 左上座標
            w, h: 幅と高さ

        Returns:
            トリミングされた画像
        """
        return image[y:y+h, x:x+w].copy()

    def upscale_image(self, image: np.ndarray, target_size: int = 128) -> np.ndarray:
        """
        AI超解像による画像拡大

        Args:
            image: 入力画像
            target_size: 目標サイズ（128x128）

        Returns:
            拡大された画像
        """
        h, w = image.shape[:2]

        if self.upscaler_available and self.upscaler is not None:
            try:
                # Real-ESRGANによる超解像
                # まず4倍に拡大してから目標サイズにリサイズ
                output, _ = self.upscaler.enhance(image, outscale=4)

                # 目標サイズにリサイズ
                result = cv2.resize(output, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                return result

            except Exception as e:
                print(f"超解像処理エラー: {e}")
                # フォールバック: 高品質なリサイズ
                return self._high_quality_resize(image, target_size)
        else:
            # Real-ESRGANが使えない場合は高品質リサイズ
            return self._high_quality_resize(image, target_size)

    def _high_quality_resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        高品質な画像リサイズ（フォールバック用）

        Args:
            image: 入力画像
            target_size: 目標サイズ

        Returns:
            リサイズされた画像
        """
        # Pillowを使用した高品質リサイズ
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # LANCZOS補間で高品質リサイズ
        resized = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # OpenCV形式に戻す
        return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

    def save_image(self, image: np.ndarray, path: str, format: str = 'PNG', quality: int = 95) -> bool:
        """
        画像を保存

        Args:
            image: 保存する画像
            path: 保存先パス
            format: 'PNG' または 'JPEG'
            quality: JPEG品質（0-100）

        Returns:
            成功した場合True
        """
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
        """
        画像を読み込み

        Args:
            path: 画像ファイルパス

        Returns:
            読み込んだ画像、失敗時はNone
        """
        try:
            # 日本語パス対応
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"読み込みエラー: {e}")
            return None
