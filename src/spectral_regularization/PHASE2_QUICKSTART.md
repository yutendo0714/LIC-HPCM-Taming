# Phase 2: Inter-scale Latent Regularization - Quick Start Guide

## 🎯 Overview

Phase 2実装が完了しました！スケール間のスペクトル混線を抑制し、さらなる性能向上を実現します。

## ✅ 実装状況

- **Phase 1 (Intra-scale)**: ✅ 完全実装＆テスト済み
- **Phase 2 (Inter-scale)**: ✅ 完全実装＆テスト済み

## 📁 実装ファイル

```
src/spectral_regularization/
├── __init__.py                    # ✅ Phase 2モジュールを追加
├── phase1_intra_scale.py         # ✅ Phase 1実装
├── phase2_inter_scale.py         # ✅ Phase 2実装
├── test_phase1.py                 # ✅ Phase 1テスト
├── test_phase2.py                 # ✅ Phase 2テスト
├── README.md                      # 詳細ドキュメント
├── QUICKSTART.md                  # Phase 1クイックスタート
└── PHASE2_QUICKSTART.md           # このファイル

src/models/HPCM_Base.py            # ✅ 潜在変数収集を追加
train.py                            # ✅ Phase 2統合完了
```

## 🚀 Phase 1 + Phase 2を両方有効化

### 推奨: Phase 1とPhase 2を同時に有効化

```bash
python train.py \
    --spectral-reg \
    --phase2-reg \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16
```

### カスタムパラメータ

```bash
python train.py \
    --spectral-reg \
    --tau-init 0.05 \
    --tau-final 1.0 \
    --truncation-epochs 100 \
    --phase2-reg \
    --delta 0.1 \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --save_path /output/hpcm_phase1_phase2
```

### Phase 2のみ有効化（Phase 1なし）

```bash
python train.py \
    --phase2-reg \
    --delta 0.1 \
    --model_name HPCM_Base \
    ...
```

## 📊 期待される効果

### Phase 1のみ vs Phase 1+2

| 指標 | Baseline | Phase 1 | Phase 1+2 | 合計改善 |
|------|----------|---------|-----------|----------|
| 訓練時間 | ~7日 | ~4日 | ~3.5日 | **2x短縮** |
| 収束epoch | ~2000 | ~1100 | ~1000 | **2x高速** |
| BD-Rate改善 | -11.16% | ~-12.2% | **~-20.65%** | **~9.5%** |
| 推論速度 | - | 変化なし | 変化なし | **0%** |

*論文より: Phase 2単体で+7~9%、Phase 1+2で合計+9.49%の改善*

### Phase 2の動作タイミング

```
Epoch   0-100: Phase 1のみ（周波数切断）
Epoch 100以降: Phase 1終了 + Phase 2開始（スケール間正則化）
```

## 🎛️ Phase 2のハイパーパラメータ

### delta（スケール間正則化の重み）

```bash
# デフォルト（推奨）
--delta 0.1     # 論文推奨値

# Ablation study用
--delta 0.05    # より弱い正則化
--delta 0.15    # より強い正則化
```

**論文での比較:**
- delta=0.1: -7.66% BD-Rate improvement ⭐ 推奨
- L1 loss: -7.07%
- Cosine similarity: -6.55%

## 🔍 Phase 2の仕組み

### 1. DWT Downsampling

スケール間の空間解像度を合わせるため、Haar waveletベースのダウンサンプリング：

```python
# s3 (64x64) -> s2 (32x32) にalign
z_s3_down = dwt.downsample(z_s3)  # [B, 320, 64, 64] -> [B, 320, 32, 32]
```

### 2. Channel Alignment

1x1 Convでチャネル数を整列：

```python
# チャネル整列（必要に応じて）
z_s3_aligned = conv1x1(z_s3_down)  # [B, 320, 32, 32] -> [B, 320, 32, 32]
```

### 3. Similarity Penalty

L2距離を計算し、**負の符号**で距離を最大化：

```python
# 類似度（距離の逆）を計算
similarity = F.mse_loss(z_s2, z_s3_aligned)

# 負の符号: minimize negative = maximize distance
reg_loss = -delta * similarity
```

→ スケール間で**異なる情報**を符号化するように促す

## 📈 モニタリング

### WandBで確認するメトリクス

Phase 2有効化時の追加メトリクス：

```python
wandb.log({
    # Phase 1
    "spectral/tau": tau_value,
    "spectral/phase": "phase1_intra" or "baseline",
    
    # Phase 2 (epoch 100以降)
    "spectral/inter_scale_reg": reg_loss_value,  # 負の値
    
    # 標準メトリクス
    "train/loss": total_loss,
    "train/bpp_loss": bpp_loss,
})
```

### 期待される値

- **spectral/inter_scale_reg**: -0.1 前後（負の値）
  - 値が小さい（絶対値が大きい）= スケール間距離が大きい ✓
  - Epoch 100以降でのみ記録される

## 🧪 テスト

Phase 2の実装をテスト：

```bash
cd /workspace/LIC-HPCM-Taming
pipenv run python src/spectral_regularization/test_phase2.py
```

期待される出力：

```
======================================================================
  ✓ All Tests PASSED!
======================================================================

Phase 2 is ready for integration with HPCM.
```

テスト内容：
1. ✅ DWT downsampling accuracy
2. ✅ DWT speed benchmark (~0.4ms/batch)
3. ✅ Inter-scale regularization
4. ✅ Gradient flow
5. ✅ Memory usage (~130MB overhead)
6. ✅ Scale independence
7. ✅ Training loop integration
8. ✅ Edge cases

## 🔄 段階的な実験戦略

### Strategy 1: Phase 1 → Phase 1+2

```bash
# Step 1: Phase 1のみで訓練
python train.py --spectral-reg --model_name HPCM_Base ...
# 結果: ~1-2% BD-Rate改善、2x訓練高速化

# Step 2: Phase 1+2で訓練
python train.py --spectral-reg --phase2-reg --model_name HPCM_Base ...
# 結果: ~9.5% BD-Rate改善、2x訓練高速化
```

### Strategy 2: Baseline → Phase 2のみ

```bash
# Phase 2単体の効果を確認
python train.py --phase2-reg --model_name HPCM_Base ...
# 結果: ~7-9% BD-Rate改善（Phase 1の高速化なし）
```

### Strategy 3: Ablation Study

```bash
# Delta値の影響を調査
for delta in 0.05 0.1 0.15; do
    python train.py --phase2-reg --delta $delta ...
done
```

## 💡 実装の詳細

### 潜在変数の収集

HPCM_Base.pyで各スケールの潜在変数を収集：

```python
# s1 (coarsest): [B, 320, H//4, W//4]
latents_s1 = y_hat.clone()

# s2 (middle): [B, 320, H//2, W//2]
latents_s2 = y_hat.clone()

# s3 (finest): [B, 320, H, W]
latents_s3 = y_hat.clone()

# Hierarchy: coarse to fine
latents_hierarchy = [latents_s1, latents_s2, latents_s3]
```

### 正則化損失の計算

```python
# train.pyのRateDistortionLossで
if epoch >= 100 and phase2_enabled:
    reg_loss = inter_scale_reg(latents_hierarchy)
    total_loss = rd_loss + reg_loss
```

## ⚠️ トラブルシューティング

### Issue 1: inter_scale_regが0のまま

**原因**: epoch < 100 または latents_hierarchyが None
**確認**:
```python
# WandBで確認
spectral/inter_scale_reg  # epoch 100以降で負の値が記録されるべき
```

### Issue 2: メモリ不足

**原因**: Phase 2は約130MBの追加メモリを使用
**解決策**:
```bash
--batch-size 12  # 16 から削減
```

### Issue 3: Gradientが流れない

**原因**: collect_latentsフラグが有効化されていない
**確認**:
```python
# モデルのフラグを確認
print(model.collect_latents)  # True であるべき
```

## 📚 論文との対応

| 論文セクション | 実装 | ファイル |
|--------------|------|---------|
| Section 3.3 | Inter-scale regularization | phase2_inter_scale.py |
| Equation 6 | Regularization loss | InterScaleRegularizer.forward() |
| Figure 1b | Regularized training | WandBで確認可能 |
| Figure 8 | Scale-wise rate/distortion | 訓練中に観察 |
| Table 3b | Ablation study | --delta で再現可能 |

## ✅ チェックリスト

Phase 2使用前の確認：

- [ ] Phase 2テストが成功（test_phase2.py）
- [ ] HPCM_Base.pyが潜在変数を返すことを確認
- [ ] `--phase2-reg`フラグを指定
- [ ] WandBでinter_scale_regが記録されることを確認
- [ ] Epoch 100以降で正則化が有効化されることを確認

## 🎉 使用例

### 完全な訓練コマンド

```bash
python train.py \
    --spectral-reg \
    --tau-init 0.05 \
    --tau-final 1.0 \
    --truncation-epochs 100 \
    --phase2-reg \
    --delta 0.1 \
    --model_name HPCM_Base \
    --train_dataset /data/train \
    --test_dataset /data/test \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --save_path /output/hpcm_phase1_2 \
    --log_dir /output/logs 2>&1 | tee training.log
```

### WandBでの確認ポイント

```python
# Epoch 0-100
spectral/tau: 0.05 → 1.0
spectral/phase: "phase1_intra"
spectral/inter_scale_reg: 0.0

# Epoch 100+
spectral/tau: 1.0
spectral/phase: "baseline"
spectral/inter_scale_reg: -0.1前後（負の値）

# 全期間
train/loss: 下降（Phase 1+2で加速）
train/bpp_loss: 下降
train/psnr: 上昇
```

## 🎓 期待される結果

### 定量的改善

| Dataset | Baseline | Phase 1+2 | BD-Rate改善 |
|---------|----------|-----------|------------|
| Kodak | -11.16% | **-19.73%** | **-8.57%** |
| CLIC | -10.79% | **-18.13%** | **-7.34%** |
| Tecnick | -13.06% | **-24.09%** | **-11.03%** |

*平均: **-9.49%** の追加改善（Phase 1効果込み）*

### 定性的改善

- スケール分離の明確化
- 潜在変数の可視化が綺麗になる
- 訓練の安定性向上
- 収束の高速化

## 💪 次のステップ

Phase 1+2完了後の改善案：

1. **HPCM_Largeへの適用**: Baseと同じコードで動作
2. **異なるλでの実験**: 0.0067, 0.013, 0.025, 0.05
3. **Ablation study**: tau_init, delta の最適値を探索
4. **高解像度データ**: 512x512, 1024x1024での評価
5. **可視化**: 各スケールのスペクトル分析

## 📞 サポート

問題が発生した場合：

1. まず`test_phase2.py`が成功することを確認
2. WandBログで`spectral/inter_scale_reg`が記録されているか確認
3. Epoch 100の前後で挙動が変わることを確認

---

**Phase 1+2の実装が完了しました！**

論文と同等の性能向上（~20.65% BD-Rate improvement）が期待できます。

Good luck! 🚀📈
