# 2026DRL-Hw3

This repository contains the code and reports for 2026 DRL Homework 3. 

## Structure Overview
- `env/`: Environment files (`GridBoard.py`, `Gridworld.py`).
- `report/`: Understanding and analysis reports for each section.
- `result/`: Training loss graphs and result images.

---

## HW3-1: Naive DQN for static mode [30%]
**【作業要求】**
* Run the provided code naive or Experience buffer reply
* Chat with ChatGPT about the code to clarify your understanding
* Submit a short understanding report
* Includes: 
  * Basic DQN implementation for an easy environment
  * Experience Replay Buffer

**【實作檔案與結果】**
* **Code:** `hw3_1main.py`
* **Report:** [HW3-1 Understanding Report](report/HW3-1_Understanding_Report.md)
* **Results:**
  * Naive DQN Loss:
    ![DQN Loss Naive](result/dqn_loss_naive.png)
  * DQN with Experience Replay Buffer Loss:
    ![DQN Loss Buffer](result/dqn_loss_buffer.png)

## HW3-2: Enhanced DQN Variants for player mode [40%]
**【作業要求】**
* Implement and compare the following:
  * Double DQN
  * Dueling DQN
* Focus on how they improve upon the basic DQN approach.

**【實作檔案與結果】**
* **Code:** `hw3_2_Double_dqn.py`, `hw3_2_Dueling_DQN.py`
* **Report:** [HW3-2 Enhanced DQN Report](report/HW3-2_Enhanced_DQN_Report.md)
* **Results:**
  * Double DQN Loss:
    ![Double DQN Loss Buffer](result/loss_double_dqn_buffer.png)
  * Dueling DDQN Loss:
    ![Dueling DDQN Loss](result/loss_dueling_ddqn.png)

## HW3-3: Enhance DQN for random mode WITH Training Tips [30%]
**【作業要求】**
* Convert the DQN model from PyTorch to either:
  * Keras, or
  * PyTorch Lightning
* Bonus points for integrating training techniques to stabilize/improve learning (e.g., gradient clipping, learning rate scheduling, etc.)

**【實作檔案與結果】**
* **Code:** `hw3_3_DQN_random.py`

## HW3-4 (加分題): Rainbow DQN
**【作業要求】**
* 使用 Rainbow DQN 解 Random Mode GridWorld
* 先分析，再教你怎麼做

**【實作檔案與結果】**
* **Code:** `hw3_4_rainbow_dqn.py`
* **Report:** [HW3-4 Rainbow DQN Report](report/HW3-4_Rainbow_DQN_Report.md)
