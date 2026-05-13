### HW3-4: Final Report & Comparison

本報告為 2026 DRL HW3 之最終總結，彙整並比較了在 Gridworld 環境中測試的各種 DQN 方法。

#### 1. 總結發現與洞見 (Findings & Insights)
在本作業中，我們從最基礎的 Naive DQN 逐步推演至整合多種機制的 Rainbow DQN：
* **經驗回放是穩定的基石**：單純使用 Naive DQN 在更新時容易受連續資料的高相關性影響而發散；加入 Experience Replay Buffer 後，訓練的穩定度與收斂成功率均有顯著提升。
* **結構改良解決估計偏差**：DQN 天生存在的過度估計問題，可透過 Double DQN 的「雙網路解耦」有效抑制；而 Dueling DQN 的「雙分支架構」則更細膩地捕捉了狀態本身的價值。
* **複雜環境需要綜合性解法**：在 Static Mode 中，基礎 DQN 即可應付；但轉移到具有高度不確定性的 Random Mode 時，單一方法的優勢逐漸受限，唯有仰賴整合性的 Rainbow DQN 才能保持優異的學習表現。

#### 2. 各方法綜合比較 (Comparison of Tested Methods)

| 演算法 (Method) | 核心改進點 | 收斂穩定性 | 運算複雜度 | 適用情境分析 |
| :--- | :--- | :---: | :---: | :--- |
| **Naive DQN** | (無, 僅基礎神經網路) | 低 | 最低 | 僅限極為簡單、靜態的小型環境。 |
| **DQN + Buffer** | 打破資料時間相關性 (i.i.d.) | 中 | 低 | 可應付一般靜態環境，但存在過度估計。 |
| **Double DQN** | 動作選擇與評估解耦 | 高 | 中 | 有效減少 Overestimation Bias。 |
| **Dueling DQN** | 拆分 V(s) 與 A(s, a) | 高 | 中 | 狀態價值高於特定動作選擇的環境。 |
| **Rainbow DQN** | 綜合上述等多項機制 | 最高 | 最高 | 複雜、隨機性高的動態環境，效能最佳。 |
