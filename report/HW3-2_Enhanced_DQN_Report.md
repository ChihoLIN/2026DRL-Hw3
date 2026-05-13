### HW3-2: Enhanced DQN Variants for player mode

本部分旨在比較與探討兩種進階的 DQN 變體：**Double DQN** 與 **Dueling DQN**，並說明它們如何改善基礎 DQN 的缺點。

#### 1. Double DQN (雙重 DQN)
在基礎的 DQN 中，目標 Q 值的計算依賴於 `max` 操作（即找出下一個狀態的最大 Q 值），這容易導致**過度估計 (Overestimation Bias)** 的問題。
* **改善機制：** Double DQN 透過將「動作選擇 (Action Selection)」與「價值評估 (Value Estimation)」解耦來解決此問題。它使用當前的線上網路 (Online Network) 來選擇下一個狀態的最佳動作，再使用目標網路 (Target Network) 來評估該動作的 Q 值。
* **優勢：** 大幅降低了 Q 值的過度估計，使神經網路的訓練過程更為穩定，策略收斂也更加準確。

#### 2. Dueling DQN (對決 DQN)
基礎 DQN 直接輸出每個 (狀態, 動作) 對的 Q 值，但在許多狀態下，選擇什麼動作其實對最終結果影響不大，真正重要的是「狀態本身的價值」。
* **改善機制：** Dueling DQN 改變了神經網路的架構，將全連接層分為兩個獨立的分支：
  1. **狀態價值分支 (State Value Function, V(s))**：評估處於當前狀態的優劣。
  2. **優勢分支 (Advantage Function, A(s, a))**：評估在當前狀態下，選擇各個動作的相對優勢。
  最後再將這兩個分支結合起來，計算出最終的 Q 值：$Q(s, a) = V(s) + A(s, a) - \text{mean}(A(s, a))$。
* **優勢：** 網路可以在不嘗試所有動作的情況下，優先學習到某些狀態本身就是好的（或壞的），進而加速學習，特別適合動作選擇不影響環境狀態的場景。
