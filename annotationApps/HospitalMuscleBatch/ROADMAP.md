# 自動セグメンテーション・ワークフロー改善ロードマップ

現在の「1件ずつ処理・確認」というフローから、「一括処理 → 一括確認・修正」という効率的なワークフローへの移行計画です。

---

## 1. 現状の課題と目標

### 現状 (As-Is)
- フォルダ選択 → 待機 → 結果確認 → 保存 → 次のフォルダ選択
- 待ち時間が細切れに発生し、作業効率が悪い
- 修正機能がないため、誤検出時は手動修正が困難

### 目標 (To-Be)
- **Batch Processing**: 複数のフォルダを一括で選択し、夜間や休憩中にまとめて予測を実行。
- **Review Mode**: 予測完了後、リストから患者を選択して高速に確認。
- **Correction**: その場でブラシツール等を使って修正し、面積・体積を即座に再計算。
- **Feedback Loop**: 修正済みデータを学習用データとして保存。

---

## 2. 実装フェーズ計画

### ✅ Phase 1: バッチ処理機能の実装 (Batch Processing) — **完了**

- [x] 親フォルダ選択 → Data.txt 解析 → シリーズ一覧生成
- [x] 複数シリーズの連続推論（QThread バックグラウンド処理）
- [x] エラーハンドリング（1件失敗しても止まらない、error_log.txt に記録）
- [x] 全体進捗バー
- [x] CUDA 失敗時の CPU 自動リトライ
- [x] デバイス選択コンボボックス（cuda / cpu）
- [x] シリーズフィルタ入力（デフォルト: eT1W_SE_cor）
- [x] 推論済み NIfTI キャッシュの活用（再推論スキップ）
- [x] NIfTI あり・CSV 行なし時の CSV 自動補完

> **補足**: BatchManager クラスの独立は行わず、MuscleSegmentationGUI 内にインライン実装。

---

### ✅ Phase 2: レビューモードとナビゲーション (Review & Navigation) — **完了（一部未実装）**

- [x] バッチ完了後に患者リスト表示（サイドバー）
- [x] クリックで患者切替・画像・体積・面積の即時更新
- [x] ○ / ✓ / △ アイコンによるステータス色分け表示
- [x] ✓ 確認 / ✎ 修正ボタンでレビューステータス設定 → 次の未判定へ自動移動
- [x] ステータスを CSV に即時書き込み（PermissionError 時はダイアログ通知）
- [x] 画像上のラベル名表示 ON/OFF
- [x] ズームスライダー（120〜200%、デフォルト 130%）
- [x] マウスホイールでスライス切替
- [ ] 「次の患者」「前の患者」キーボードショートカット（未実装）
- [ ] 元画像とオーバーレイの高速切り替え（現状はラベル名の表示/非表示のみ）

---

### 🔲 Phase 3: インタラクティブ修正ツール (Segmentation Editor) — **未実装・次フェーズ**

既存の **EyeMuscleAnnotationPortable (AnnotationTool)** に近づける形で実装予定。

- [ ] ブラシ（Brush）/ 消しゴム（Eraser）ツール
  - AnnotationTool の `AnnotationCanvas` を参考に、`ScrollableImageLabel` を拡張
  - マウスドラッグで `pred_array` を直接書き換え
  - ブラシサイズ変更スライダー
- [ ] 編集対象ラベル（筋肉名）選択
- [ ] 修正確定時に断面積（Area）・体積（Volume）を再計算し CSV 更新
  - `NnUNetPredictor.visualize_slice` の計算ロジックを独立関数に切り出し
- [ ] Undo（修正の取り消し）

> **設計方針**: AnnotationTool の CanvasWidget + ポリゴン編集のUIパターンを参考に、
> ピクセルレベルの編集（ブラシ）として再実装する。

---

### 🔲 Phase 4: 学習データへのフィードバック (Data Feedback) — **未実装・将来**

- [ ] 修正後マスクを `Corrected_Data/` に別名保存
- [ ] nnU-Net 学習データ形式（`Case_XXXX.nii.gz`）への自動変換・保存オプション
- [ ] メタデータ（修正者・修正日時）の記録

---

## 3. 実装済み追加機能（ロードマップ外）

以下は当初計画にはなかったが実装済みの機能。

| 機能 | 内容 |
|---|---|
| 体積 3 方式 | 全スライス・動的範囲・固定範囲 (5〜11) の 3 タブ表示 |
| 動的範囲自動検出 | so/sr/mr/lr/ir が両側揃ったスライスを自動特定 |
| L/R 放射線規則対応 | 画像左 = 患者右 (`r_`)、画像右 = 患者左 (`l_`) |
| 操作一覧ポップアップ | AnnotationTool と同様の ❓ ボタン + QMessageBox |
| ズームスライダー制御 | 体積タブのホイールスクロール無効化 |

---

## 4. 技術的な実装詳細

### 現在のクラス構成

| クラス | 役割 |
|---|---|
| `NnUNetPredictor` | 推論・可視化・体積計算（全 3 方式） |
| `PredictionThread` | 新規推論の QThread |
| `LoadExistingThread` | キャッシュ再読み込みの QThread |
| `MuscleSegmentationGUI` | メインウィンドウ・バッチ管理・レビュー管理 |
| `ScrollableImageLabel` | ホイールイベントをシグナル化する QLabel |
| `NoScrollTabWidget` | ホイールスクロール無効の QTabWidget |
| `ResultManager` | NIfTI / CSV の保存・読み込み・ステータス更新 |

### Phase 3 に向けた拡張方針

- `ScrollableImageLabel` → `SegmentationEditorLabel` に拡張（マウスドラッグでブラシ描画）
- `NnUNetPredictor.visualize_slice` の面積計算部分を `compute_areas(pred_slice, spacing)` として独立させ、編集後の即時再計算に対応
- 修正確定 → `ResultManager.save_prediction_nifti` で上書き & `append_to_csv` / `update_review_status` を呼び出し

### データフロー（現状）

1. **入力**: 親フォルダ選択 → Data.txt 解析 → batch_queue
2. **処理**: DICOM → NIfTI 変換 → キャッシュ確認 → 推論 or キャッシュ読み込み
3. **保存**: NIfTI 書き出し → CSV 追記（行がなければ補完）
4. **確認**: ユーザーがリストからクリック → 画像・体積表示 → ✓/✎ でステータス確定

---

## 5. 今後の優先順位

1. **Phase 3（修正ツール）のプロトタイプ** — AnnotationTool を参考にブラシ編集の最小実装
2. **Phase 2 の残り** — キーボードショートカット（次/前の患者）
3. **Phase 4（学習フィードバック）** — Phase 3 完了後
