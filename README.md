# LLMの仕組みをゼロから学ぶ

numpy だけを使って LLM（大規模言語モデル）の内部動作を実際の数値計算で理解するノートブック集です。

## 構成

| ノートブック | 内容 |
|------------|------|
| `01_tokenization_embedding.ipynb` | テキスト → トークン → 埋め込みベクトル → 位置エンコーディング |
| `02_self_attention.ipynb` | Q/K/V 行列 → Scaled Dot-Product Attention → Multi-Head Attention |
| `03_transformer_block.ipynb` | Layer Norm + FFN + 残差接続 → N 層スタック → LM Head |
| `04_training_inference.ipynb` | Cross-Entropy 損失 → 勾配 → パラメータ更新 → 自己回帰生成 |

## 特徴

- **ライブラリは numpy のみ** — PyTorch 不要で動く
- **小さい数値で全部見える** — `d_model=4`, `seq_len=4` など最小サイズで行列を表示
- **手計算の確認付き** — 各ステップを 1 スカラーずつ手で追える
- **4 ノートブックが繋がる** — NB01 の出力が NB02 の入力になる一貫した設計

## 実行方法

```bash
pip install numpy jupyter
jupyter notebook
```

ブラウザで各ノートブックを開き、上から順にセルを実行してください。

## 前提知識

- Python の基礎（リスト, 辞書, クラス）
- numpy の基礎（配列操作, `@` 演算子）
- 高校数学レベルの線形代数（行列積, 転置）

## 計算の流れ

```
テキスト
  ↓ [NB01] トークン化・埋め込み
X = Embedding(tokens) + PositionalEncoding
  ↓ [NB02] Self-Attention
Q, K, V = X @ W_{Q,K,V}
attn = softmax(Q @ K.T / √d_k) @ V
  ↓ [NB03] Transformer ブロック
LayerNorm(X + MHA(X)) → LayerNorm(X' + FFN(X'))
logits = X_out @ W_lm
  ↓ [NB04] 学習・推論
loss = -log P(正解)  →  W ← W - lr * ∇W
next_token = argmax(softmax(logits[-1]))
```

## 参考文献

- Vaswani et al. (2017) "Attention Is All You Need"
- Radford et al. (2018) "Improving Language Understanding by Generative Pre-Training"
- Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
