### HW3-1 Understanding Report: Naive DQN & Environment Interaction

#### 1. 環境建構與狀態表示 (Environment & State Representation)

本作業的環境由 `GridBoard.py` (底層資料結構) 與 `Gridworld.py` (物理引擎與邏輯) 共同構成，負責定義馬可夫決策過程 (MDP) 中的核心要素：

* **狀態表示 (State Representation)：** 為了讓神經網路能有效提取空間特徵，`GridBoard.render_np()` 採用了類似影像通道 (Channels) 的「空間 One-hot 編碼」。它將 $4 \times 4$ 的二維棋盤，轉換為形狀為 `(4, 4, 4)` 的三維張量 (Tensor)，將 Player、Goal、Pit 與 Wall 精確分離在不同的矩陣圖層中。
* **動作與防呆機制 (Action & Validation)：** `Gridworld.makeMove()` 定義了四個離散的動作空間 (上下左右)。環境內建 `validateMove()` 防呆機制，若代理人 (Agent) 的動作會導致撞牆或超出邊界，則判定為無效移動並將代理人保留在原地。
* **獎勵函數 (Reward Function)：** 透過 `Gridworld.reward()` 給予環境回饋。代理人抵達目標 (Goal) 獲得 $1$ 分，誤踩陷阱 (Pit) 獲得 $-1$ 分，其餘安全移動為 $0$ 分。

#### 2. 基礎 DQN 實作邏輯 (Basic DQN Implementation)

Naive DQN 使用深度神經網路取代傳統的 Q-table，作為 Q 值的函數逼近器 (Function Approximator)。

* **神經網路架構：** 模型接收上述 `(4, 4, 4)` 的狀態張量作為輸入，並在輸出層給出該狀態下四個可能動作的預期 Q 值 (Q-value)。
* **損失函數 (Loss Function)：** 訓練目標為最小化預測 Q 值與目標 Q 值之間的均方誤差 (MSE Loss)。目標值 $y$ 的計算基於貝爾曼方程式。透過計算 TD Error 並進行反向傳播 (Backpropagation) 更新網路權重 $\theta$。在 `static` 模式下，因環境物件完全固定，模型僅需記憶並擬合出一條固定的避障路徑即可順利收斂。

#### 3. 經驗回放機制 (Experience Replay Buffer)

若直接使用代理人與環境互動的連續資料進行訓練，會因資料具備高度「時間相關性 (Temporal Correlation)」而導致神經網路訓練極不穩定甚至發散。

* **資料儲存：** 實作 Experience Replay Buffer，將代理人每一步的探索經驗打包成 $(S, A, R, S', Done)$ 的多元組形式存入雙向佇列中。
* **隨機抽樣 (Random Sampling)：** 在進行神經網路權重更新時，從 Buffer 中隨機抽取一小批 (Mini-batch) 的歷史資料進行學習。此機制成功打破了資料間的連續性，確保訓練資料更符合獨立同分配 (i.i.d.) 的假設，從而大幅提升 DQN 在複雜環境下學習的穩定性。
