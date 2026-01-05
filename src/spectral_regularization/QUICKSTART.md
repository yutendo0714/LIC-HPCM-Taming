# Phase 1: Intra-scale Frequency Regularization - Quick Start Guide

## 🎯 Overview

Phase 1実装が完了しました！このガイドでは、すぐに訓練を開始する方法を説明します。

## ✅ 実装状況

- **Phase 1 (Intra-scale)**: ✅ 完全実装＆テスト済み
- **Phase 2 (Inter-scale)**: 🚧 未実装（次のステップ）

## 📁 実装ファイル

```
src/spectral_regularization/
├── __init__.py                    # モジュール初期化
├── phase1_intra_scale.py         # ✅ DCT truncation実装
├── test_phase1.py                 # ✅ テストスイート
├── README.md                      # 詳細ドキュメント
└── QUICKSTART.md                  # このファイル

train.py                            # ✅ Phase 1統合済み
```

## 🚀 すぐに始める

### Step 1: 実装の検証

```bash
# テストを実行
cd /workspace/LIC-HPCM-Taming
pipenv run python src/spectral_regularization/test_phase1.py
```

✅ **期待される出力:**
```
✓ All Tests PASSED!
Phase 1 is ready for training.
```

### Step 2: 訓練開始

#### 基本的な使い方（Phase 1有効化）

```bash
python train.py \
    --spectral-reg \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16
```

#### カスタムパラメータ（推奨）

```bash
python train.py \
    --spectral-reg \
    --tau-init 0.05 \
    --tau-final 1.0 \
    --truncation-epochs 100 \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --save_path /output/hpcm_phase1
```

#### Baselineとの比較用（Phase 1なし）

```bash
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --save_path /output/hpcm_baseline
```

### Step 3: 進捗モニタリング

WandBで以下のメトリクスを確認：

1. **spectral/tau**: 0.05 → 1.0 へ線形増加（最初の100 epoch）
2. **train/loss**: Baselineより早い収束を確認
3. **spectral/phase**: "phase1_intra" → "baseline" への切り替え

## 📊 期待される効果

### 訓練効率

| 指標 | Baseline | Phase 1 | 改善 |
|------|----------|---------|------|
| 収束epoch | ~2000 | ~1100 | **1.8x高速** |
| 訓練時間 | ~7日 | ~4日 | **2x短縮** |

### 性能向上

- **BD-Rate改善**: 約1-2%の追加改善（Phase 2でさらに7-9%向上）
- **収束安定性**: Lossカーブが滑らか
- **推論コスト**: ゼロ増加（訓練時のみの正則化）

## 🎛️ ハイパーパラメータ調整

### tau_init（初期周波数cutoff）

```bash
# デフォルト（推奨）
--tau-init 0.05    # 5%の周波数から開始

# Ablation study用
--tau-init 0.025   # より保守的（遅いが安定）
--tau-init 0.1     # より積極的（速いがリスク）
```

**論文での比較（Table 3a）:**
- 0.025→1.0: 1.62x speedup, -1.01% BD-Rate
- **0.05→1.0**: 1.84x speedup, -1.07% BD-Rate ⭐ 推奨
- 0.1→1.0: 1.77x speedup, -1.05% BD-Rate

### truncation_epochs（適用期間）

```bash
# デフォルト（推奨）
--truncation-epochs 100

# より長期間適用
--truncation-epochs 150   # より慎重な学習

# より短期間
--truncation-epochs 75    # 速い切り替え
```

## 🔍 トラブルシューティング

### Issue 1: メモリ不足

**症状**: CUDA out of memory
**解決策**:
```bash
# Batch sizeを削減
--batch-size 12  # または8

# Patch sizeを削減
--patch-size 256 256  # デフォルトから変更なし、必要なら128
```

### Issue 2: 収束が遅い

**症状**: 100 epoch後もLossが高い
**チェック項目**:
1. `spectral/tau`が正しく増加しているか確認
2. `--tau-init`を小さくしてみる（0.05→0.025）
3. Learning rateスケジュールを確認

### Issue 3: NaNやInf

**症状**: 訓練中にNaN/Infが発生
**解決策**:
```bash
# Gradient clippingを強化
--clip_max_norm 0.5  # デフォルトは1.0

# 入力データの範囲を確認
# DCT前後で[-5, 5]程度が正常
```

## 📈 可視化

### 生成される可視化ファイル

テスト実行時に生成されます：

```bash
test_outputs/
├── frequency_truncation_progression.png  # 周波数切断の進行
└── radial_masks.png                      # 各tauでのマスク
```

### WandBグラフの推奨設定

**重要なメトリクス:**
1. `spectral/tau` vs epoch（線形増加を確認）
2. `train/loss` vs epoch（Baselineと比較）
3. `train/bpp_loss` vs epoch
4. `train/psnr` vs epoch

## 🔄 次のステップ

### Phase 2の準備

Phase 2（Inter-scale regularization）を実装する際に必要な準備：

1. **潜在変数の収集**: HPCMモデルが各スケールの潜在変数を返すように修正
2. **DWTモジュール**: PyWavelets (`pywt`) のインストール
3. **スケール間alignment**: Conv1x1モジュールの追加

詳細は今後のPhase 2実装ガイドで説明します。

## 📝 実験ログのテンプレート

```yaml
# experiment_config.yaml
experiment_name: "HPCM_Base_Phase1"
model: "HPCM_Base"
lambda: 0.013

spectral_reg:
  enabled: true
  tau_init: 0.05
  tau_final: 1.0
  truncation_epochs: 100

training:
  epochs: 3000
  batch_size: 16
  learning_rate: 5e-5
  patch_size: [256, 256]

expected_results:
  convergence_epoch: ~1100
  training_time: ~4 days
  bd_rate_improvement: ~1-2%
```

## ✅ チェックリスト

実験開始前の確認事項：

- [ ] テストが全て成功（test_phase1.py）
- [ ] WandBプロジェクトが設定済み
- [ ] 訓練データとテストデータのパスが正しい
- [ ] GPUメモリが十分（16GB推奨）
- [ ] `--spectral-reg`フラグを指定
- [ ] Baseline実験も並行して実行（比較用）

## 🎓 論文との対応

実装と論文のセクションの対応：

- **Section 3.2**: Intra-scale regularization → `phase1_intra_scale.py`
- **Equation 3**: DCT transform → `DCTTransform.dct_2d()`
- **Equation 4**: Radial mask → `create_radial_mask()`
- **Equation 5**: IDCT transform → `DCTTransform.idct_2d()`
- **Figure 1a/1b**: Training dynamics → WandBでモニタリング
- **Table 3a**: Ablation study → `--tau-init`の変更で再現可能

## 💡 Tips

### 効率的な実験管理

```bash
# 複数のlambdaで実験
for lambda in 0.0067 0.013 0.025 0.05; do
    python train.py \
        --spectral-reg \
        --model_name HPCM_Base \
        --lambda $lambda \
        --save_path /output/phase1_lambda_${lambda} &
done
```

### ログの保存

```bash
# 詳細ログを保存
python train.py --spectral-reg ... 2>&1 | tee training_phase1.log
```

### 途中からの再開

```bash
python train.py \
    --spectral-reg \
    --checkpoint /output/epoch_500.pth.tar \
    ...
```

## 📞 サポート

問題が発生した場合：

1. まず`test_phase1.py`が成功することを確認
2. WandBログで`spectral/tau`が正しく記録されているか確認
3. Baseline（Phase 1なし）と比較して異常がないか確認

## 🎉 始めましょう！

全ての準備が整いました。Phase 1の訓練を開始してください：

```bash
python train.py --spectral-reg --model_name HPCM_Base ...
```

Good luck! 📈
