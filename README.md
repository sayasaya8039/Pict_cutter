# Pict Cutter - 画像トリミング＆超解像アプリ

画像ファイルを読み込み、AI余白検知による自動トリミングと高品質な拡大処理を行うWindowsアプリケーションです。

## 機能

- **画像読み込み**: ファイル選択またはドラッグ＆ドロップで画像を読み込み
- **AI余白検知**: OpenCVによるエッジ検出で自動的にコンテンツ領域を検出
- **1:1トリミング**: 正方形の選択範囲で画像をトリミング
- **手動選択**: マウスドラッグで任意の範囲を選択可能
- **高品質拡大**: 128x128などの任意サイズに高品質リサイズ
- **保存**: PNG/JPEG形式で保存（JPEG品質調整可能）

## 対応フォーマット

- 入力: PNG, JPEG, BMP, GIF, WebP
- 出力: PNG, JPEG

## 使い方

1. 「画像を開く」ボタンで画像を読み込む（またはドラッグ＆ドロップ）
2. 「自動検出」ボタンでAIが余白を検知し、最適な正方形領域を選択
3. 必要に応じてマウスドラッグで選択範囲を調整
4. 出力サイズを設定（デフォルト128px）
5. 「トリミング＆拡大」ボタンで処理を実行
6. プレビューを確認し、「保存」ボタンでファイル出力

## ビルド方法

### 必要環境

- Python 3.10以上
- Windows 10/11

### セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# 開発時の実行
python src/main.py

# EXEビルド
pyinstaller --onefile --windowed --name "Pict_cutter" --distpath "Pict_cutter" src/main.py
```

## 技術スタック

- **GUI**: PySide6 (Qt for Python)
- **画像処理**: OpenCV, Pillow
- **余白検知**: Canny Edge Detection + Contour Detection
- **高品質リサイズ**: LANCZOS補間

## ライセンス

MIT License
