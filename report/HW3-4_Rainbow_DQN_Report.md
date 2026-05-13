### HW3-3: Rainbow DQN for random mode

本部分探討如何透過整合性的 **Rainbow DQN** 方法，在更具挑戰性的隨機模式 (Random Mode) 中提升代理人的表現。

#### 1. Rainbow DQN 概念整合
Rainbow DQN 並非單一的創新演算法，而是將數個針對 DQN 提出的獨立優化機制進行整合，以發揮其綜效。通常包含以下技術的結合（視實作內容而定）：
* **Double DQN**：解決 Q 值過度估計。
* **Dueling DQN**：分離狀態價值與動作優勢，提升學習效率。
* **Prioritized Experience Replay (PER)**：優先從 Buffer 中抽取 TD Error 較大（學習空間較大）的經驗，而非均勻隨機抽樣。
* **Multi-step Learning**：使用多步回報 (N-step return) 加速獎勵訊號的傳遞。
* **Distributional RL** 與 **Noisy Nets** 等進階機制。

#### 2. 多重優化機制的綜合影響 (Impact on Performance)
在「隨機模式」下，環境的動態性更高（例如目標或障礙物位置可能改變），單純的 DQN 往往面臨探索效率低、收斂困難的挑戰。
透過結合上述改進方法：
1. **穩定性增強**：Double 與 Dueling 架構讓估計值更準確，減少策略震盪。
2. **樣本效率提升**：結合 PER 或多步學習，代理人能更有效率地利用稀疏的獎勵訊號進行學習。
3. **魯棒性 (Robustness)**：Rainbow DQN 能在複雜或隨機性高的環境中，展現出遠勝於基礎 DQN 的抗干擾能力與最終效能表現。
