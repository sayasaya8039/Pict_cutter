# Icon Gene Form

アイコン生成＆トリミングに特化したWindowsデスクトップアプリケーション

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## 概要

Icon Gene Formは、画像のトリミングとAI画像生成機能を備えたアプリケーションです。SNSアイコンやアプリアイコンの作成に最適化されています。

## 主な機能

### 画像トリミング
- **AI自動検出**: OpenCVによるエッジ検出で最適な1:1領域を自動選択
- **3段階の検出感度**: ゆるい / 普通 / きつい から選択可能
- **ウォーターマーク除外**: 四隅のロゴ・透かしを自動的に検出から除外
- **手動選択**: マウスドラッグで任意の正方形領域を選択
- **ドラッグ＆ドロップ**: ファイルを直接ウィンドウにドロップして読み込み

### AI画像生成
- **Google AI Studio連携**: Gemini APIを使用した画像生成
- **モデル選択**:
  - Gemini 2.5 Flash Image（標準・高速）
  - Gemini 3.0 Pro Image（高品質・プレビュー）
- **アイコン最適化**: プロンプトを自動的にアイコンスタイルに調整

### 出力設定
- **サイズ選択**: 32×32 / 64×64 / 128×128 / 256×256 / 512×512 / 1024×1024
- **高品質リサイズ**: LANCZOS補間による美麗な拡大処理
- **保存形式**: PNG / JPEG / ICO（品質1-100調整可能）
- **保存先記憶**: アプリ起動中は最後に保存したフォルダを記憶
- **自動ファイル名**: 元画像のファイル名をベースにした保存名を提案

## スクリーンショット

```
+------------------------------------------+
|  Icon Gene Form - アイコン生成＆トリミング  |
+------------------------------------------+
|  [画像表示エリア]      | 検出感度設定     |
|                        | 選択範囲情報     |
|  [画像を開く]          | 処理設定         |
|  [自動検出] [リセット]  | [プレビュー]     |
|  [AI画像生成] [設定]   | 保存設定         |
+------------------------------------------+
```

## 動作環境

- Windows 10 / 11
- 画面解像度: 1024×768以上推奨
- インターネット接続（AI画像生成機能使用時）

## インストール

### 実行ファイル（推奨）

1. [Releases](../../releases)から最新の`Icon_gene_form.exe`をダウンロード
2. 任意のフォルダに配置
3. ダブルクリックで起動

### ソースからビルド

#### 必要環境
- Python 3.10以上
- pip

#### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/your-username/Pict_cutter.git
cd Pict_cutter

# 依存パッケージのインストール
pip install PySide6 opencv-python numpy Pillow

# 開発時の実行
python pict_cutter.py

# EXEビルド
pyinstaller --onefile --windowed --collect-all cv2 --name Icon_gene_form --icon=icon.ico pict_cutter.py
```

## 使い方

### 基本的なトリミング

1. **画像を読み込む**
   - 「画像を開く」ボタンをクリック、またはドラッグ＆ドロップ

2. **選択範囲を設定**
   - 「自動検出」で AI が最適な範囲を提案
   - または、マウスドラッグで手動選択

3. **トリミング実行**
   - 出力サイズを選択
   - 「トリミング＆拡大」をクリック

4. **保存**
   - 形式（PNG/JPEG/ICO）と品質を設定
   - 「保存」をクリック

### AI画像生成

1. **APIキーを設定**
   - 「設定」→ Google AI Studio APIキーを入力
   - APIキーは [Google AI Studio](https://aistudio.google.com/) で取得

2. **画像を生成**
   - 「AI画像生成」をクリック
   - モデルとサイズを選択
   - プロンプト（英語推奨）を入力
   - 「生成」をクリック

3. **生成画像を使用**
   - 「この画像を使用」で編集画面に読み込み

## 検出感度について

| 設定 | 説明 | 用途 |
|------|------|------|
| ゆるい | 余白を多めに残して広く検出 | 背景を含めたい場合 |
| 普通 | バランスの取れた検出（デフォルト） | 一般的な用途 |
| きつい | 余白を最小限にタイト検出 | 被写体のみ切り出し |

## 対応フォーマット

| 種別 | 形式 |
|------|------|
| 入力 | PNG, JPEG, BMP, GIF, WebP |
| 出力 | PNG, JPEG, ICO |

## 技術スタック

- **GUI**: PySide6 (Qt for Python)
- **画像処理**: OpenCV, Pillow, NumPy
- **AI画像生成**: Google Gemini API
- **余白検知**: Canny Edge Detection + Contour Detection
- **高品質リサイズ**: LANCZOS補間
- **ビルド**: PyInstaller

## ファイル構成

```
Pict_cutter/
├── pict_cutter.py        # メインアプリケーション
├── icon.ico              # アプリケーションアイコン
├── requirements.txt      # 依存パッケージ
├── README.md             # 開発者向けドキュメント
├── Icon_gene_form.spec   # PyInstaller設定
└── Icon_gene_form/
    └── Icon_gene_form.exe  # 実行ファイル
```

## トラブルシューティング

### アプリが起動しない
- Windows Defender やアンチウイルスソフトの例外設定に追加

### AI画像生成でエラーが発生
- APIキーが正しく設定されているか確認
- インターネット接続を確認
- API利用制限に達していないか確認

### 自動検出がうまく機能しない
- 検出感度を変更して試す
- 手動選択を使用

## 更新履歴

### v1.1.0 (2025-12-18)
- ICO形式での保存に対応
- 保存先フォルダをアプリ起動中記憶する機能を追加
- 保存時のファイル名を元画像ベースに変更

### v1.0.0 (2025-12)
- 初回リリース
- 画像トリミング機能
- AI自動検出機能（3段階感度）
- 四隅ウォーターマーク除外
- AI画像生成機能（Gemini API）
- PNG/JPEG保存機能
- 複数出力サイズ対応

## ライセンス

MIT License

## 作者

Created with Claude Code

## 貢献

バグ報告や機能要望は Issue でお知らせください。
