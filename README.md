# E-Divisive

同一個資產在不同的時間序列下會有不同的統計分佈，彼此的差異由歐式距離來衡量，能量距離定義如下

$$
\hat{D}^2 = \frac{2}{nm} \sum_{i=1}^{n} \sum_{j=1}^{m} |x_i - y_j| - \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j| - \frac{1}{m^2} \sum_{i=1}^{m} \sum_{j=1}^{m} |y_i - y_j|
$$

X區間和Y區間如果是同分布的，距離就應該等於0


## 尋找最佳分割點的機制

二分法 (Bisection)與迭代搜尋來定位分割點：

* **最大化準則 (Argmax)**：
    它會在整個時間序列中移動分割位置 $\tau$，計算在該位置下左側序列 $X_{\tau}$ 與右側序列 $Y_{\tau}$ 之間的 $\hat{\mathcal{Q}}$ 值。  
    **最佳分割點 $\hat{\tau}$** 就是讓 $\hat{\mathcal{Q}}$ 達到最大值的那個時間點(distinct distributions) ：
* **引入修正變數 $\kappa$**：
    為了避免在存在多個變動點時產生混淆，演算法允許右側區間的長度 $\kappa$ 是變動的。這樣可以確保在搜尋單一分割點時，不會因為後方還有其他分佈變化而干擾當前的估計。

$$(\hat{\tau}, \hat{\kappa}) = \{argmax}_{(\tau, \kappa)} \hat{\mathcal{Q}}(\mathbf{X}_{\tau}, \mathbf{Y}_{\tau}(\kappa); \alpha)$$

alpha 這裡是敏感度參數，取值介於0-2


* **Permutation Test**:
每找到一個潛在轉折點，演算法會進行Permutation Test，如果 p-value 大於設定的門檻（0.05），則停止繼續分割 。
迭代次數 R 設定依照樣本長度斟酌設置


# 範例
**Cisco金融數據**
<img width="1160" height="367" alt="image" src="https://github.com/user-attachments/assets/82b4e79a-1241-43f2-8855-1dff53e05bd4" />

**Squared returns ACF**

<img width="1283" height="334" alt="image" src="https://github.com/user-attachments/assets/77ced0e0-f38b-4db4-b163-ec83ea82269b" />


**MU**
<img width="1552" height="447" alt="image" src="https://github.com/user-attachments/assets/121aac3e-ffc5-4007-b057-0132969d5e96" />

<img width="1577" height="486" alt="image" src="https://github.com/user-attachments/assets/7ed5da46-e0f5-492a-a99b-5dd61fbc8366" />

# 結論
基本上分割後的樣本內波動符合獨立特性，波動叢聚現象不存在，波動叢聚是數據結構隨時間改變的產物

算法成功精確的分辨Regime Shifts

論文參考: A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data
