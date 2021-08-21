# Machine Learning 2021

### Chapter 1 : Introduction of Machine / Deep Learning

#### Different types of Functions
:::info
機器學習目的 : 讓機器具備找一個函式的能力
:::
例如做語音辨識(丟入一段語音讓機器辨識)、圖像辨識、或者是下棋,\
這些東西都可以利用找到一個函式,丟入一筆資料後輸出答案。\
較常用的幾種函式類型如**Regression**(輸出一個scalar)、**Classification**(輸出一個類別),也可以讓機器產生一個有結構的物件(Structured Learning)

#### How to find a function
1. **Function with Unknown Parameters** : 猜一個function, 假設要找的函式是一個Linear的Function, 我們可以將函式猜成 $y = bias + weight \times x_1$ , 其中$weight 跟 bias$要透過學習來找到最好的一組。
2. **Define Loss from Training Data** : 機器找到一個$function$後要評定其的好壞, 所以我們有$loss$。$loss$會透過實際值($\hat{y}$)跟預測值($y$)來計算。簡單來說, $loss$**就是用來看預測值跟實際值差多少**。\begin{gather}Loss = \cfrac{1}{N}\sum_{n}  e_n, \\e_n = error\ function(y, \hat{y}),\\ error\ function\::MAE, MSE \end{gather}
3. **Optimization** : 機器有Loss function後要讓訓練結果不斷逼近實際值(找最小的Loss) \begin{gather} w^1 \leftarrow w^0 - \eta \cfrac{\partial L}{\partial w}|_{w = w^0}\\ \end{gather} 其中$\eta = learning\ rate$, 為$\mbox{hyperparameters}$, 需要自己訂的值,也需要自己訂的值還有$\mbox{batch_size, epoch}$。透過這個步驟不斷迭代更新$weight 跟bias$。但因為是看$\mbox{gradient}$找最小值, 不見得可以找到$\mbox{global minimum}$

:::warning
模型參數的修改來自對問題的理解
:::

#### while model is too simple
有時候要找的function不見得是Linear,\
如果是這樣,我們用Linear的function無論怎麼逼近都沒有辦法找到最好的答案。\
這種時候我們就要給Model更多的彈性(點取得夠多就跟原本的function越像)\
**Sigmoid Function**:
\begin{align}
y &= c\ \cfrac{1}{1+e^{-b + wx_1}}\\
& = c\ sigmoid(b + wx_1)\\
\end{align}這個函式可以把原本直線變成圓滑的曲線\
也可以針對每一個$x$找不同的$weight 跟 bias$
\begin{align}
&原本 : y = b + wx_1\\
&新的 : y = b + \sum_{i} c_i\ sigmoid(b_i + \sum_{j} w_{ij}\ x_j)\\
\end{align}
當然,還是要回歸到機器學習最一開始的步驟, 去計算**Loss**、去進行**Optimization**\
通常這種學習方法一次不是只讀一筆資料, 而是用**batch**去計算\
而一個**epoch**則是讀完全部資料一次。

**ReLU**:
\begin{gather}
y = c\ max(0, b + wx_1)
\end{gather}

:::warning
像$\mbox{Sigmoid}$或是$\mbox{ReLU}$這種都是$\mbox{Activation function}$,\
可以讓模型找到的函式更逼近實際的函式
:::

#### Deep Learning
前面講的都是針對一個layer去做計算,\
我們也可以疊多一點的layer, 讓參數的數量進一步變多,當然擬合的效果也會更好一點。\
這種疊多個$\mbox{hidden layer}$的學習方式就叫$\mbox{Deep Learning}$
:::warning
$\mbox{hidden layer}$可以讓擬合的效果更好,但也會造成"過擬合"(**Overfitting**),\
使模型訓練出來沒有辦法對類似狀況作出反應,\
這個部分在後續章節還會講解
:::

### Chapter 2 : Overfitting

#### Review : Framework of ML
![](https://i.imgur.com/WhQ7TJl.jpg)
$\mbox{Step 1: 找一個包含未知數的function}$\
$\mbox{Step 2: 定義loss function}$\
$\mbox{Step 3: 找一組參數讓他的loss是最小的}$

#### Model Bias
如果在訓練的過程中,**train data**的loss降不下來,可能有兩個問題:\
1. **model bias** : 代表model太簡單了, 需要把你的model變有彈性一點
2. **Optimization** : 下一章會提到

想要將model變得有彈性, 最主要有兩種方式(**大海撈針,但針不在海裡**):
1. 增加feature
2. 增加Deep learning的層數

如果是Optimization的問題, \
那可能代表這個演算法沒辦法找到model的最低loss(**大海撈針, 針在海裡但找不到**)
:::warning
怎麼辨別是Model Bias還是 Optimzation Issue
:::

#### Model Bias v.s. Optimization Issue
如果增加了layer, 但訓練結果卻更爛了,就代表是Optimization的問題。\
這時候代表你需要找到更好的演算法去擬合這個case。

#### Overfitting
如果今天發生的是**train data的loss很低, 但在test data的loss卻很高**\
這種情況有可能是model"**太彈性了**",造成test的效果不好。\
**Solution**:
1. **More train data(增加資料量)** : 利用資料來限制模型的彈性
2. **Data augmentation(資料增強)** : 利用一些不會改變圖片答案的變化隨機加在圖片上,例如 : 水平翻轉、放大縮小、物件平移等等的操作來,但要做合理的增強(做有可能會出現的增強)![](https://i.imgur.com/VcWIJvM.jpg)
3. **Make model simple** : 可以透過減少模型參數(或層數)來達成讓模型簡單一點,減少overfitting發生。
4. **Early stopping** : 在模型已經達到test loss訓練最低的時候就停下訓練過程, 實作步驟可以參考功課內容。
5. **Regularization** : 對loss動手腳, 讓他在高次項出現時進行懲罰,避免model太彈性,參考功課實作。
6. **Dropout** : 拋棄一部分的參數, 參考功課實作。
7. **Cross Validation** : 從訓練資料抽取一部分來做預測, 透過這個方式知道模型訓練過程是否發生過擬合。

#### Mismatch
Mismatch跟overfitting不同, **Mismatch指的是資料的domain不同**,\
如 : 丟照片訓練,卻丟卡通圖片測試(後面章節會講到如何解決這個問題)

### Chapter 3 : Small Gradient
在訓練過程中, 因為是透過gradient來進行迭代, 有可能造成被卡在local minimum的問題, 這時候參數沒辦法繼續更新。(gradient = 0)

#### Math part
**Tayler Series Approximation :**
\begin{gather}
L(\theta) \approx L(\theta\ ') + (\theta - \theta\ ')^Tg + \cfrac{1}{2}(\theta - \theta\ ')^TH(\theta - \theta\ ')
\end{gather}
**Gradient g** is a **Vector :**\
\begin{gather}
g = \nabla L(\theta\ '),   g_i = \cfrac{\partial L(\theta\ ' )}{\partial \theta _i}
\end{gather}
**Hessian H** is a **Matrix :**
\begin{gather}
H_{ij} = \cfrac{\partial ^2}{\partial \theta _i \partial \theta _j}L(\theta \ ')
\end{gather}

當在**critical point**的時候, **gradient = 0**, 剩下的**Hessian** 可以用來判斷附近的地貌。

**At critical point :**\
\begin{align}
L(\theta) &\approx L(\theta\ ') + \cfrac{1}{2}(\theta - \theta\ ')^TH(\theta - \theta\ ')\\
&\approx L(\theta\ ') +\cfrac{1}{2} v^THv
\end{align}
1. $v^THv > 0$ : \
**H is positive definite = All eigen values are positive** = $\underline{\mbox{Local minimum}}$
2. $v^THv < 0$ : \
**H is negative definite = All eigen values are negative** = $\underline{\mbox{Local maximum}}$
3. Sometime $v^THv > 0$, Sometime $v^THv < 0$ : \
**Some eigen values are positive and some are negative** = $\underline{\mbox{Saddle point}}$

遇到Saddle point時, 沿著eigen value的方向去更新參數就能讓loss變小,\
實際上, local minimum較不常出現, Saddle point 比較常出現。
:::warning
運算量太大了, 實際上沒有在用
:::

#### Review: Optimization with Batch
把資料分成一個批量一個批量(batch)去做訓練。\
**epoch** : 看完全部的batch(看完全部資料一次)\
**Shuffle** : 隨機打亂資料, 讓每個epoch看到的batch都不一樣。

#### Small batch v.s. Large batch
![](https://i.imgur.com/SpaPVIa.jpg)

因為平行運算, 只要在上限內, 無論batch大小,時間都會差不多。\
因此考慮平行運算後, 可以發現batch大小差別只在一個資料更新比較慢, 一個資料更新比較快, 差別只有在訓練過程是否比較Noisy, 又Noisy對訓練其實是有幫助的。\
**結論 : small batch is good for testing, and large batch is bad for testing**\

Original paper:[On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)、[Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes](https://arxiv.org/abs/1711.04325)、[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)、[Stochastic Weight Averaging in Parallel: Large-Batch Training that Generalizes Well](https://arxiv.org/abs/2001.02312)、[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)、[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
#### Momentum
在真實的物理世界中, 物體向下滑落不會只單純考慮坡的斜度, 可能還會有其他現象, 我們把這些現象加到Gradient Descent裡面來實作。

**Vanilla Gradient Descent**\
隨著參數不斷更新, 不斷減小前進的力道(有點類似摩擦力)。\
就機器學習的角度上來說, 隨參數更新也會不斷地接近Loss最低的位置, 這時候learning rate也需要隨之變小, 才有辦法更逼近最低點。

**Momentum**
每一部移動前, 考慮前一步的移動方向(類似慣性。)

#### Concluding Remarks
* Critical points have zero gradients
* Critical points can be either saddle points or local minima, which can be determined by Hessian matrix. 
    * It is possible to escape saddle points along the direction of eigenvectors of Hessian matrix.
    * Local minima may be rare
* Smaller batch size and momentum help escape critical points

### Chapter 4 : Optimizer

#### Adaptive Learning Rate
因為error surface太崎嶇了, 我們需要針對每一個參數都訂定不同的learning rate。\
有時候當loss停止更新了, 不見得代表你遇到了critical point,\
有時候單純就只是你卡住了, gradient並沒有變得很小。\
在實際上訓練的過程中很少卡到critical points,\
因為一般的gradient descent會在gradient還很大的時候就被卡住了。
因此我們需要更好的Optimizer。
#### Root Mean Square
\begin{aligned}
&\theta_i^{t+1} \leftarrow \theta^t_i - \cfrac{\eta}{\sigma^t_i}g^t_i\\
&\sigma^t_i = \sqrt{\cfrac{1}{t + 1}\sum^t_{i = 0}(g^t_i)^2}\\
& g^t_i = \cfrac{\partial L}{\partial \theta_i}|_{\theta = \theta^t}
\end{aligned}
利用這個方法, 讓gradient descent在進行迭代的時候多考慮整個訓練過程的斜率。
:::warning
**Used in Adagrad**
:::

#### RMSProp
**Adagrad問題**\
就算是同一個參數, 需要的learning rate也會隨著時間改變,就算是同一個參數、同一個方向, 也希望learning rate可以動態調整。\
(Adagrad考慮了整路上的gradient限制了Optimizer的彈性)\
**RMSProp**\
\begin{aligned}
&\theta_i^{t+1} \leftarrow \theta^t_i - \cfrac{\eta}{\sigma^t_i}g^t_i\\
&/*從原先adagrad公式改變\sigma算法*/ \\
&\sigma^t_i = \sqrt{\alpha(\sigma^{t-1}_i)^2 + (1 - \alpha)(g^t_i)^2}\\
&/*只考慮最近的兩個, 並且用\alpha來調整哪個比較重要*/
\end{aligned}
**結論 : 遇到斜率改變時, Adagrad的反應會比RMSProp慢**

#### Adam
其實就是 RMSProp + Momentum, 通常裡面的參數不用做特別調整就可以得到不錯的結果了。\

Original paper : [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf)

#### Learning Rate Scheduling
![](https://i.imgur.com/oc65CMQ.jpg)
沒有調整的情況下, 有太多小的y方向的gradient, 走到gradient比較大的地方就會慢慢震盪回來。可以透過**Learning Rate Scheduling**調整回來。\
**Learning Rate Scheduling  : 隨參數更新次數增多, 降低$\eta$**

**Warmup**\
分成兩個部分, 一開始把learning rate調大, 達到高點後再把learning rate調小。\
會多出三個hyperparameter : 
1. 變大要變多大
2. 變大多快
3. 變小多快

合理的解釋為, 一般在訓練開始的時候$\sigma^t_i$的variance比較大, 所以需要比較大的learning rate去做。\
Original paper : [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)、[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

### Chapter 5 : Classification
Classification就是分類問題, 給機器一個輸入, 輸出他屬於哪一類。\
如果class間沒有明確的關係, 不能用class 1、class 2表示, 必須要用one-hot vector表示

#### Softmax
把y的domain挪到0~1。
\begin{gather}
y_i' = \cfrac{exp(y_i)}{\Sigma_jexp(y_i)}\\
1 > y_i' > 0\ \ and\ \  \Sigma_iy_i' = 1
\end{gather}
只有兩個class時, softmax跟sigmoid是一樣的。\
總結一下softmax的作用 : 
1. Normalized 到 0~1
2. 讓所有值的和為1
3. 讓差距更大

#### Cross-entropy(Loss of Classification)
\begin{aligned}
&e = -\sum_i \hat{y_i}lny_i'\\
&L = \cfrac{1}{N}\sum_n e_n 
\end{aligned}
通常Cross-entropy會跟Softmax綁再一起用,\
**Minimizing cross-entropy 就相等於 Maximizing likelihood**
:::warning
相較於MSE, Cross-Entropy在分類問題的loss差距比較明顯, 有利迭代更新參數。
:::

### Chapter 6 : Normalization
效果就是可以把本來很崎嶇的error surface炸平。
\begin{gather}
\tilde{x}^r_i \leftarrow \cfrac{x^r_i - \mu_i}{\sigma_i}
\end{gather}
做完Normalized後, 所有dimension的mean會變成 0, Variance會變成 1。\
透過feature normalization, 可以讓gradient descent收斂更快。

#### Considering Deep Learning
在activation function前做或是後做的差異其實並不大。\
做feature normalized後, 所有值都會一起變動, 但實際上實作只會針對batch\
(GPU不會一次讀入所有資料去計算)。
:::warning
batch_size必須要夠大做起來才有意義
:::
Original paper : [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)\
To learn more : [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)、[Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/abs/1702.03275)、[Layer Normalization](https://arxiv.org/abs/1607.06450)、[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)、[Group Normalization](https://arxiv.org/abs/1803.08494)、[Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)、[Spectral Norm Regularization for Improving the Generalizability of Deep Learning](https://arxiv.org/abs/1705.10941)

### Chapter 7 : Convolutional Neural Network(CNN)
![](https://i.imgur.com/l2k9QNc.jpg)
通常圖片都會有三個通道(RGB)\
隨參數越多, 模型越彈性, 越容易overfitting

**Observation 1 :** Identifying some critical pateterns\
(要看出是什麼東西只要看某些重點就好了, 例如可以看有沒有鳥喙判斷是不是鳥)\
不一定要看整張圖片才能辨識出東西, 有一些pattern比圖片小很多
![](https://i.imgur.com/JAQGCRc.jpg)
透過Observation 1來設計的模型, 透過挪動kernel去找尋圖片中的pattern\
**Observation 2 :** The same patterns appear in different regions\
(相同物件可能出現在圖上不同地方)
原本的作法沒辦法達到參數共享, 但我們必須讓receptive field彼此互相共享參數。![](https://i.imgur.com/o7oZgak.jpg)

#### Convolutional Layer
一一去比對跟目標物件的相似度, 並給出一個相似度分數,\
每次向右移動的距離可以事先設定(stride)。\
透過這樣子操作, 我們可以算出一個**Feature Map**![](https://i.imgur.com/OcZ4Dqb.jpg)


#### Maxpooling
**Observation 3 :** Subsampling the pixels will not change the object\
(把圖片做Subsampling, 並不會讓人類對圖片的認知出現異常)
![](https://i.imgur.com/TEVffHB.jpg)
把前面Convolutional Layer做出來的Feature Map分組, 每一組取最大的出來, 要幾個一組可以事先決定。

:::warning
Convolutiona Layer跟Maxpooling可以重複出現、交替使用。
也可以不做pooling避免小的特徵被丟掉
:::

#### The whole CNN
在前面的Convolution Layer跟Maxpooling重複數次之後, 再經過Flatten和Fully-Connected Layers, 就是完整的CNN
* Flatten : 把參數拉直, 變成一長串的一維矩陣

#### Application
* **Playing go :** 下棋的部分, 因為棋盤的大小固定為19x19, 刪除掉任何一個棋盤上的資訊就會變成不同狀況, 所以沒有辦法做pooling(Observation 3 不成立)
* **Speech recognize :** 語音的部分因為receptive field不能共用, 要對網路再做修正。
:::warning
如果用在其他領域, 要針對其特性再去做修正
:::

### Chapter 8 : Self Attention
前面討論的Model都是輸入一個vector, output一個Class或是一個Scalar, 那如果我們今天輸入的是一長串的vector要如何處理(如何處理更複雜的輸入)。


#### Vector Set as Input
![](https://i.imgur.com/fmR7Rok.jpg)
設定一個Windows, 每次挪動一點去讀完整個語音訊息, 就可以用來處理類似的輸入。
![](https://i.imgur.com/VYAQWxX.jpg)
Graph也可以是一組vector, 可以把每個節點想像成是一個vector。

#### What is the output
1. **Each vector has a label(這裡著重於此) :** 輸入一個vector就輸出一個label。為了讓機器可以考慮上下文, 我們會讓機器的windows大一點, 如果機器一個字一個字讀的話, 可能會有問題(例如saw在同一個句子裡可能同時出現名詞和動詞的意思); 但也不能蓋住全部句子, 如果蓋住整個sequence, 句子有長有短的情況下容易overfitting。
2. **The whole sequence has a label :** 一整組資料只有一個label。
3. **Model decides the number of labels itself :** 讓機器自己決定

#### Self-attention
![](https://i.imgur.com/OJJzFNM.jpg)
每個結果都是考慮整個Sequence才得到的。\
可以把Fully-Connected Network跟Self-attention交替使用。\
Original paper : [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**實作方式**\
假設輸入一組vector(裡面包含$a_1, a_2 \dots$), 會得到一組vector(裡面包含$b_1, b_2 \dots$)
![](https://i.imgur.com/R2aW59O.jpg)

1. 每個$a$分別產生一個$query$和一個$key$, 每個$query$跟所有產生的$key$做一次內積(計算關聯性), 之後經過一個**Softmax Layer**(也可以換成其他的)
2. 每個$a$再產生一個$v$, 去對產生出來的內積相乘。
3. 將剛剛跟$v$產生的結果總合起來, 得到$b$
\begin{aligned}
&b^1 = \sum_i \alpha'_{1,i}v^i\\
&\alpha'_{1,i} = \cfrac{exp(\alpha_{1,i})}{\sum_j exp(\alpha_{1,j})}
\end{aligned}

可以利用這個方式, 把所有的$b$產生出來。\
這一串操作其實是一連串的矩陣乘法。
:::warning
每個結果是獨立的, **不需要依次產生**
:::

#### Multi-head Self-attention
關聯性的計算方式不只一種, 定義也不只一種, 所以也許不能只有一組參數。\
如果要一次產生多組參數, 則代表每個$a$必須要產生多$q、k$跟$v$, 而每個產生$q、k、v$的$a$就是一個head。

#### Position Encoding
在Self-attention中, 節點在哪個位置對演算法沒有影響, 這樣的設計對原本的命題有問題, 需要多考慮位置參數。(可以透過人為設定(hand-crafted)或是公式產生)\
Original paper : [Learning to Encode Position for Transformer with Continuous Dynamical Model](https://arxiv.org/abs/2003.09229)
#### Self-attention v.s. CNN
圖片也可以當作是一組vector, (一個貫穿所有channel的pixel當作一個vector), 但是CNN是以kernel為單位, 所以CNN其實是Self-attention的簡化版。\
Original paper : [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)、[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)、[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584)、 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
#### Self-attention v.s. RNN
RNN跟Self-attention處理的問題類似, 但是RNN有前後的關聯, 最前面的參數可能帶不到後面, 相較之下Self-attention比較有效率一點。\
Original paper : [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)

#### 結語
Self-attention的變形很多, 都叫xxformer。\
但每每看到有加快運算速度的method出來, 都會發現performance變差。\
Original paper : [
Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006)、[Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
:::warning
依據不同的問題, 可以再對模型做一點調整
:::

### Chapter 9 : Transformer(Seq2seq)
輸出的長度由輸入控制(例如 : 翻譯)。\
也可以透過語音辨識 + 文字翻譯達成目的, 但有些語言是沒有文字的。(閩南語)

#### Question Answering(QA)
大部分的自然語言處理其實都是透過問答題來達成的。\
利用給機器問題, 希望訓練機器回答正確答案。\
也可以設計選擇題甚至是多選題給機器去學習, 那這時候模型就要在收到輸入後自己決定有幾個答案。
其實這樣的學習方式, 就是把輸入的Sequence丟到encoder內, 透過計算後由decoder輸出一個Sequence。

#### Encoder
![](https://i.imgur.com/9Mo1Ozx.jpg)
因為可能會有未知的位置資訊(參考前一張Self-attention說明), 進入後經過Multi-Head Self-attention和Fully-Connected的Layer運算後得到結果,在Fully-Connected Layer可以使用CNN或是RNN來運算。\
這樣的block可能會重複數次。

#### Decoder
Decoder從Encoder拿到參數後, 透過運算之後為每一個可能出現的文字給上分數, 通過softmax層後, 把分數最高的那個當作答案輸出出去。\
Decoder的特點是會把自己的輸出當作下一部的輸入, 但只要中間錯了一個字, 後面的答案就會錯光光。
![](https://i.imgur.com/5zyjvMr.jpg)
仔細看會發現Decoder的長相跟Encoder有一點像, 差別就是這個Masked Multi-Head Attention的Layer。

**Masked Self-Attention**
![](https://i.imgur.com/tb5E8DM.jpg)
因為Decoder的產生是有順序的, 所以要想像後面的key還沒被產生出來。\

**Autoregressive(AT)**
一次產生一個字, 機器自動去生成下一個可能會出現的字。
輸出跟輸入的長度關係複雜, 希望機器可以自己學習怎麼樣去判斷長度,\
而實際上如果只有這樣, 機器不知道輸出的長度是多少, 他會永無止盡的一直預測下一個字, 我們必須再多做一點設定(Stop Token), 就像前面的Start Token一樣。\
當預測結果是Stop Token後就代表句子結束了。\


**Non-autoregressive(NAT)**
簡單來說就是一次把整個句子產生出來, 而不是像上面一樣, 一次產生一個字。
相較來說, 因為平行運算的關係, 速度相較AT快了不少, 而且可以控制輸出的長度。但Performance不及AT(因為Multi-modality的關係)

#### Transformer
![](https://i.imgur.com/AoQ9VQ9.jpg)

**Cross-Attention**
![](https://i.imgur.com/Bm40DiC.jpg)
從Decoder產生query, 從Encoder的輸出產生key跟v, 作法如圖上所示\
其實是先有Cross-Attention才有Self-Attention的出現。

**Training**
![](https://i.imgur.com/QURTBFF.jpg)
從上面這張圖可以發現, 在訓練過程中其實是有答案提供給Decoder看的。\
當然在訓練上也有一些小訣竅

**Tips**\
有一些東西是不需要自己產生的, 例如xxx你好, 只需要把xxx複製下來就好, 複製遠比自己產生向名字這種跟前後沒有關係的字簡單的多。


### Chapter 10 : Generative Adversarial Network(GAN)
generate是另外一種我們希望可以讓機器學會的東西, 也就是讓機器自己去產生一些聲音、圖片或是一段文字。而產生一張圖片對機器而言其實也不過就是產生一個高為向量, 至於要產生多少dimension, 就讓機器自己去決定。

#### Why distribution
過去的預測方式是利用過去的資訊去判斷可能的答案, 例如如果我們讓機器人去玩遊戲, 可能可以往右走, 也可以往左走, 無論往右走跟往左走都可以說是正確答案, 但如果今天同時往左又往右走, 可能就不是我們所樂見的結果。\
而經由前面方式訓練出來的模型卻有可能產生這樣的結果。\
因此我們在模型訓練的時候加上了一個隨機的數值, 使得機器隨機輸出, 避免出現兩面討好的結果, 而當機器需要創造力的時候, 就會需要隨機分布。


#### Discriminator
Discriminator可以想像成是一個分類器, 由自己設計的Neural Network組成, 他會嘗試分辨出哪些是實際上的圖片, 哪些是機器產生的圖片。

#### Basic Idea of GAN
有了這些, 我們可以大致定義一下要怎麼樣來訓練一個GAN。
![](https://i.imgur.com/SF1o8gA.jpg)
利用交叉訓練Generator跟Discriminator, 來讓生成的圖片越來越接近真實的圖片, 彼此透過彼此的輸出來調整訓練。

**Algorithm**
1. Fix(固定) generator G, and update discriminator D
2. Fix discriminator D and update generator G(這個步驟generator會嘗試去混淆discriminator, 讓他以為generator產生的圖片是真正存在的圖片)

利用不斷重複步驟一跟步驟二去訓練GAN的模型。

#### Theory behind GAN
:::info
**訓練目標\
Generator : Maximize Discriminator Accuracy\
Discriminator : Minimize Classification Loss\
簡單來說, 希望輸出的圖片跟真實的圖片差別越小越好**
:::

\begin{gather}
G^*_{(loss)} = arg\ \mathop{min}_G Div(P_G, P_{data})
\end{gather}
意即希望$P_G$ 跟 $P_{data}$的差別越小越好

**How to compute the divergence**\
其實沒有想像中那麼複雜, 只要可以sample data出來就可以算了。\
分別從$P_{data}$跟$P_G$sample一些資料出來, 即便我們可能不知道他們的分布, 我們還是可以算他們的$Divergence$。

**Discriminator**\
\begin{gather}
D^* = arg \mathop{max}_D V(D, G)
\end{gather}

Discriminator的任務是要清楚地分出真實的圖跟生成的圖, 這中間的$V(D,G)$其實是negative cross entropy, 也就是實際上要train的是希望cross entropy越小越好。
:::warning
算divergence的方式有很多種, 也可以選擇用別種divergence來計算。
但即便是這樣, GAN還是一個不好train的模型
:::

#### JS divergence is not suitable
1. 在實際訓練下, $P_{data}$跟$P_G$其實是不會有overlapped的。即便有, 也可以忽略它。
2. 即便有, 只要sample的點夠多, 就能夠清楚分開。
3. JS divergence is always $log2$ if two distributions do not overlap.(在這樣的情況下, JS Divergence無法分辨好壞程度)

如果再沒有辦法判斷好壞的情況下, classifier可能會硬背。\
可以利用換一種divergence(**Wasserstein distance**)來解決這個問題。\
在Wasserstein distance中, 只要$P_{data}$跟$P_G$稍微有一點變化, 在distance上馬上就會有變化, 有這個才能做出好的generator。\
利用這個function時, 我們會希望Discriminator找到的function變化不要這麼劇烈, 要找平滑一點的function, 因為如果變化太劇烈, 很容易會有極大植跟極小值, 可以利用ReLU來解決這個問題。

#### Model Collapse
同一張臉越來越多。
換句話說, Generator抓到了Discriminator的盲點, 不斷朝那個方向發展。\
會造成Generator生成的臉來來去去都是那幾張, 多樣性不足。\
可以把進入softmax前的輸出來做一些運算, 評估生成的相似度。\
但即便GAN出現了這麼多的版本, 始終沒辦法完全解決這個問題。

#### Conditional Generation
我們希望不只產生圖片, 我們還希望可以操控Generator的輸出, 產生有特定特徵的圖片, 例如 : 紅色眼睛的動漫人物或是黃色頭髮的。
![](https://i.imgur.com/YUoZow8.jpg)
我們可以在Generator輸入時多給一個條件, 但如果只是這樣, Discriminator還是只會做把生成圖片跟真實圖片分類的工作, 這樣沒有辦法判斷說我們人類的輸入是否有被考慮進去, 所以Discriminator也要重新訓練。\
因此在訓練Discriminator也要多告訴他條件。\
利用這個方式, 我們可以把輸入的部分換成一張圖片, 藉此去改變生成圖片的domain。\
當然也可以把輸入的部分改成聲音, 那機器就會受聲音影響去產生圖片, 聲音越大聲圖片越清楚。

#### Learning from unpaired data(Cycle GAN)
有時候訓練GAN的時候, 不總是有正確答案給機器參考。\
(但或多或少還是需要成對資料才能訓練)\
可以參考Pseudo labeling和back translation的實作
![](https://i.imgur.com/uvi70zt.jpg)
簡單來說, 從$x\ domain$  sample一張圖片出來,\
訓練機器從$y\ domain$ sample一張最像的圖片出來。\
但這樣的生成方式沒有辦法用Discriminator做驗證, 因為Discriminator只會判斷輸出, 即便輸出跟輸入沒什麼關係。\
因此我們需要第二次轉換, 把原先那張$y\ domain$的照片再做一次,\
看看機器從$x\ domain$ sample的照片是不是我們原先sample的那張。
:::warning
即便做成這樣, 也無法保證生成的照片會完全一樣,\
畢竟只是在兩個domain裡面去sample照片, 也不保證是有正確答案的, 
:::
利用這種訓練方式, 可以作例如Text style Transfer(把負面的詞語轉成正面的詞語)等模型。

### Chapter 11 : Self-Supervised Learning(Bert)
Bert模型真的超級大的啦...

#### Self-supervised Learning
**Supervised**是有答案, 可以讓機器在產生出結果後對答案;\
但**Self-Supervised** 是指產生出模型後, 讓模型去label資料, 再繼續訓練下去, \
是一種upsupervised learning的方式\
訓練方法出奇的簡單, 就是訓練機器做填空題, 把部分的輸入用特殊的token蓋住, 也可以隨機替換成其他字, 希望訓練機器可以預測出被蓋住實際上是哪個字。
![](https://i.imgur.com/f5Tdmbl.jpg)
最後發現, 我們雖然只訓練機器做填空題, 但實際上這個模型可以被用在其他任務上, 不只是填空題可以用。

#### GLUE(General Language Inderstanding Evaluation)
任務 : 
1. Corpus of Linguistic Acceptability(CoLA)
2. Standford Sentiment Treebank(SST-2)
3. Microsoft Research Paraphrase Corpus(MRPC)
4. Quora Question Pairs(QQP)
5. Semantic Textual Similarity Benchmark(STS-B)
6. Multi-Genere Natural Language Inference(MNLI)
7. Question-answering NLI(QNLI)
8. Recognizing Textual Entailment(RTE)
9. Winograd NLI(WNLI)

#### How to use BERT

**Case 1 :**\
我們可以把輸入的Sequence丟入BERT中, 希望輸出一個class, \
因為BERT本身已經是train過的模型了, 效果會比隨機初始來得好。 \
例如做Sentiment analysis。\
**Case 2 :**\
輸入一個sequence, 希望輸出一個Sequence而且跟原本的長度一樣,\
我們可以在做完BERT之後, 利用BERT輸出經過一些Linear Layer, 來得到我們的答案。 \
例如做詞性標記(POS tagging)。\
**Case 3 :**\
一次輸入兩個Sequence, 希望得到一個class, 可以在兩個Sequence中間插入分隔符號。例如 : Natural Language Inferencee(NLI)。\
**Case 4 :**\
![](https://i.imgur.com/pKWU01Y.jpg)
Extraction-based Question Answering(QA)\
做問答題, 答案就在輸入裡。利用**Case 3**的方法,\
將問題跟文章分開, 在把答案推出來之前做一次Random Initialized(長度跟BERT輸出一樣), \
之後經過Softmax找出分數最高的, \
找出答案在文章中的哪個位置(從哪裡開始、在哪裡結束)。
:::warning
做BERT的時候, 可以把資料做遮住、刪除、重新排列、或是轉置等處理,\
給資料一些雜訊, 這樣效果會比較好。
:::

#### Why does BERT work
BERT在處理一段文字的時候會考慮上下文, 所以即便是同一個字(Apple)也可以找出不同的見解(手機 or 水果)。這個能力不見得跟填字有關, 說不定這個方法就剛好適合這種大型模型訓練。

### Chapter 12 : Auto-Encoder
在Self-supervised learning裡, 我們會投入大量未標註label的資料給機器去做學習,\
因為我們並未設定明確目標, 所以要讓機器自己去找目標學習\
**Auto-Encoder功能 : 決定一個不用標註資料就可以學習的任務**
:::warning
其實跟Cycle GAN的做法相似, 利用換domain之後再回推, 看是否和原輸入一樣。
:::

#### Why Auto-encoder
因為圖片的變化是有限的, 所以我們可以利用這個方法找出有限的變化, 以此來簡化輸入。

#### De-noising Auto-encoder
在進入encoder之前, 我們會將圖片加上一些雜訊, 但是要求機器還原的是沒有加雜訊的輸入。\
這個做法其實就是BERT的做法, 加上mask但是要求機器要還原原本的詞。

#### Feature Disentanglement
![](https://i.imgur.com/DYwb5RM.jpg)
Disentanglement : 把原本糾纏在一起的東西解開。\
我們希望可以讓機器在學習的階段, 就學到一些輸入中的資訊, 例如如果輸入一段聲音, 可能某些參數是在記錄內容、某些參數是在記說話的人等等。\
但在實作上, 向量確實可能包含上面這些資訊, 但那些維度代表哪些內容卻無從得知, Feature Disentanglement的概念即在train Encoder的同時知道那些維度代表那些資訊。

### Chapter 13 : Explanable Machine Learning
Explanable Machine Learning : 簡單來說就是需要機器是用什麼標準去判斷正確答案的。\
因為機器得到正確答案, 不代表機器就真的學到了正確的東西, 有些時候可能是因為人類沒有注意到的細節造成了這種錯誤, 所以我們需要具有解釋力的機器學習。\
但大部分時候, 好解釋的模型, 可能代表他的能力比較差...。\
就像在明亮處找鑰匙比較簡單, 但鑰匙不一定真的掉在明亮的地方。

#### Goal of Explanable ML
![](https://i.imgur.com/Rg1lPim.jpg)
人能接受的Explanation就是好的Explanation。\
例如如果想要在圖片中找到有貓的圖片, \
就必須告訴大家貓在哪裡, 或是從哪裡判斷是貓的。\
如果我們更動這些feature, 機器就會大幅改變他的決定, \
代表這些特徵對機器在決定上來說是很重要的。
\begin{gather}
&\{x_1, \dots, x_n, \dots , x_N\} \Rightarrow \{x_1, \dots, x_n + \Delta x, \dots , x_N\}\\
&e \Rightarrow e + \Delta e\\
&|\cfrac{\Delta e}{\Delta x}| \Rightarrow |\cfrac{\partial e}{\partial x_n}|
\end{gather}
:::warning
![](https://i.imgur.com/6g7C8n0.jpg)
Saliency Map中, 越白代表越重要
:::

#### Limitation

**Noisy Gradient**
![](https://i.imgur.com/MeopTa9.jpg)
有時候在做Saliency Map時, 圖片上可能會存在很多雜訊, 造成人類難以辨認那些東西是重要的。\
這時候可以用**SmoothGrad**來進行調整\
:::warning
**SmoothGrad : \
Randomly add noises to the input image,\
get saliency maps of the noisy images, and average them.**
:::

**Gradient Saturation**\
有時候Gradient沒辦法反映真正的重要程度, 鼻子長度在一定的長度範圍內, 也許可以直接判斷他是大象, 但超出這個範圍後影響就變得不是那麼大。

#### How a network processes the input data
經過前面的說明, 我們明白在輸入資料上的某些特徵確實會被機器給蒐集起來, 但我們不確定他在哪個位置, 那我們要怎麼樣找到他們呢?

**Principal Component Analysis(PCA) or t-SNE**\
先對要測試的維度進行資料降維(t-SNE有視覺化功能), 試著把比較複雜的維度做簡化, 方便我們理解。

**Attention**\
利用attention找關聯性的方法來進行分群(參考前面幾張的內容)

這種做法就像插一根探針進去檢查這個Layer, 再用分類器去把資料分類、視覺化給人做參考, 但可能因為參數沒調好而有誤差。

:::warning
前面的做法是Local Explanation, 簡單來說就是試著回答為什麼會得到這個答案,\
接下來的做法是Global Explanation, 簡單來說就是告訴人這個答案應該長甚麼樣子。
:::

#### Global Explanation
![](https://i.imgur.com/CipagJ8.jpg)
從CNN的模型出發, 我們可以將某個Layer抽取出來, 希望把這些參數回推成一張圖片。\
但我們必須在train這個generator時加上一些限制, 來避免機器在沒看到的情況下卻說有看到。 需要完成這個generator需要很多很多的Constraint才有辦法做到。

:::warning
也有人會利用Linear Model去模仿Neural Network,\
利用Linear Model好解釋的特性, 製作出等效的模型再去解釋。\
但只能小區域模仿。
:::

### Chapter 14 : Domain Adaptation
![](https://i.imgur.com/0J4hofc.jpg)
當訓練資料跟測試的資料分布不同, 即便模型訓練好, 也沒辦法做出正確的預測。\
本章講解如何處理domain不同的訓練方式(Transfer Learning)。\
但要這樣做, 需要對Target domain也要有一定程度的了解, 所以多少還是需要一些資料是從target domain出來而且有label的資料。\
我們希望機器在做這個類型的訓練的時候, 可以抽取一樣的部分出來, 忽略掉不一樣的部分\
#### Domain Adversarial Training
![](https://i.imgur.com/sbR9BWB.jpg)
這種訓練方法跟GAN有點像, 可以把Feature Extractor想作是Generator。\
訓練的目標是讓每個domain的差別越小越好, 甚至沒有差別, 但這個做法實際上是在分開每個domain, 訓練Discriminator(未必是最好的做法)。\
可能會有一個問題是Discrominator一直輸出0, 就會分不出來,\
但這件事情實際上不會發生,\
**因為訓練除了要滿足Discriminator, 還要滿足Label Predictor**。
:::warning
永遠都不可能完全分開兩個domain, 因為一定會有overlapping的部分,\ Domain Adaptation沒有想像中好Train
:::


### Chapter 15 : Network Attack
模型訓練出來之後的正確率高是不夠的, 還需要能應付人類的惡意。

#### How to attack
對圖片加上一些雜訊, 讓模型對圖片給出錯誤的預測, 但雜訊不能太大造成整張圖片無法辨識, 最好讓雜訊小到人類看不出來, 但是機器卻會給出錯誤的預測。\
有時候在做攻擊的時候, 我們會希望有一個固定的目標(例如要讓機器把貓看成鍵盤或海星), 或是做無差別的攻擊(不預期會預測出什麼, 但是預測錯誤)\

**Non-target**
\begin{aligned}
&x^* = arg \mathop{min}_{d(x^0, x) \leq \varepsilon} L(x)\\
&(希望讓x跟x_0的差距小於 \varepsilon, 可以假設超過 \varepsilon 人類就會看出破綻)\\
&L(x) = -e(y,\ \hat{y})\\
&(希望cross-entropy越大越好)
\end{aligned}

**Targeted**
\begin{aligned}
&L(x) = -e(y, \hat{y}) + e(y, y^{target})\\
&(除了跟答案差距要大, 還要跟目標像)
\end{aligned}

**Non-perceivable**
\begin{aligned}
&d(x^0, x) \leq \varepsilon\\
&(\varepsilon需要考慮人類感知能力, 還需要一些domain\ knowledge)\\
\end{aligned}
* L2-norm
\begin{aligned}
&d(x^0, x) = ||\Delta x ||_2\\
&=(\Delta x_1)^2 + (\Delta x_2)^2 + (\Delta x_3)^2 \dots\\
\end{aligned}
* L-infinity
\begin{aligned}
&d(x^0, x) = ||\Delta x ||_{\infty}\\
&=\mathop{max}\{|\Delta x_1|, |\Delta x_2|, |\Delta x_3|, \dots\}\\
&(\mbox{L-Infinity}較接近人類感知能力)
\end{aligned}

#### Attack Approach
更新輸入, 不更新參數。
\begin{gather}
x^* = \mathop{arg} \mathop{min}_{d(x^0, x) \leq \varepsilon} L(x)
\end{gather}
改變input, 讓他去minimize loss即可, 記得要考慮人類感知能力。

#### White Box v.s. Black Box
* White Box Attack : 需要知道模型參數
* Black Box Attack : 不知道模型參數, 需要用猜的
Black Box攻擊可以利用丟入輸入、取得輸出, 來猜出模型參數(訓練一個差不多的模型), 但相對來說, 黑箱攻擊要做Target Attack就比較難

#### Universal Adversarial Attack
同一個雜訊可以造成多個資料掛點。\
例如 : 神奇眼鏡(戴上眼鏡就會把人看成別人)、把3鼻子拉長一點就被看成8
**Adversarial Reprogramming**\
操控模型做一些不是訓練目標的事

#### Defense

**Passive Defense**\
對輸入圖片做一些輕微的模糊化處理, 就可以應對一些比較簡單的attack signal, 但也不能太過頭, 避免本來辨識對的成功率就下降了。\
也可以做圖片壓縮(降低解析度), 甚至讓Generator建出一張一模一樣的圖片。\
但這種被動防禦一旦被人知道就無效了, 還是可以有破解的方法, 隨機的效果也有限。\
:::warning
這裡說的處理都是在模型訓練好之後的test做處理。
:::

**Proactive Defense**\
在訓練的時候就自己去攻擊自己的模型, 並讓他知道會有這種狀況, 其實就是資料增強(Data Augmentation)

### Chapter 16 : Reinforcement Learning
![](https://i.imgur.com/HL7awRN.jpg)
對抗式學習常見於讓機器學會打遊戲這件事情, \
找一個Function讓機器針對環境的觀察, 做出相應的動作, 看看環境給出的回應如何。\
Reward就像是遊戲中的得分或扣分, 或是遊戲中的中止條件等等, 希望機器可以取得比較好的分數。\
還有一種是遊戲只有輸或贏(在遊戲中並沒有甚麼特定的分數, 如圍棋), 這種訓練我們就必須在訓練的過程中自己給他一些回饋, 方便機器人去學習。


**回歸到機器學習最基本的訓練方式**
![](https://i.imgur.com/EeHy6Q3.jpg)

#### Function with Unknown
得到一個來自環境的輸入, 如遊戲畫面, 可以是一張圖片或是其他東西,\
找到函式要對這個輸入進行一次分類, 類別即可以進行的操作(往左走、往右走、開火等等)\
對每個動作Network中都會提供一個機率, \
**但並不是挑最高的出來, 而是依據機率隨機選擇一個。**
:::warning
假設往左走的分數為0.7, 往右走為0.2, 開火為0.1\
則有70%機率往左走, 20%機率往右走, 10%機率開火
:::

#### Define Loss
對每一個機器所做的行為提供一個分數, 例如開火並擊殺敵人可以獲得若干分數等。\
一場遊戲從開始到結束稱為一個episode,\
reward為採取行為後可以立即得到的分數, return則是一整場遊戲的reward\
我們希望一場遊戲中的分數越高越好, 則 : 
\begin{gather}
R = \sum_{t = 1}^T r_t\\
Loss = -R\ \ \ (Loss越小越好)
\end{gather}

#### Optimization
1. Actor輸出有隨機性, 同樣一個輸入不一定可以得到同一個答案
2. Environment 跟 Reward 不是 Network, Environment也具有隨機性。

因此在這樣的條件下, optimization是最難的(隨機性),\
Reward需要看得分、Observation 跟 action

#### Policy Gradient
為了控制機器做最佳化, 無論選擇或不選擇, 都要給一個分數
\begin{aligned}
&Take\ action\ \hat{a} : L = e\\
&Don't\ take\ action\ \hat{a} : L = -e
\end{aligned}
利用類似Train Classifier的作法去控制actor的行為
\begin{gather}
L = e_1 - e_2\\
(e_1為take\ action, e_2為don't\ take\ action)\\
\theta^* = \mathop{arg} \mathop{min}_{\theta} L
\end{gather}
經過這樣的設計, 我們可以紀錄出每個行為做或不做, 以及要怎麼樣去計算它的Loss。\
但這樣的行為, 不代表好不好, 只是代表想不想執行。(有時候需要犧牲短期利益來換取背後利益)

\begin{gather}
L = \sum A_ne_n\\
\theta^* = \mathop{arg} \mathop{min}_{\theta} L\\
\end{gather}
什麼都不做對actor也是一種選項, \
有時候如果不特別訂定分數, 可能會造成actor一直甚麼都不做,\
因此要對某些特定行為做處理, 相當於是人為的希望actor去執行或不去執行。\
多考慮人類對actor的介入, 但這樣還不夠, **這樣的寫法只會讓機器考慮短期利益**\
第一個版本希望利用累積的方式(累積人類對機器的介入分數), 來影響actor做決定。\
\begin{gather}
G_t = \sum^N_{n = t} r_n
\end{gather}
但這樣的作法並不合理, **因為第一步會影響到最後一步的機會不高**
\begin{gather}
G_t' = \sum^N_{n = t} \gamma^{n-t} r_n
\end{gather}
利用第二個版本, 讓離$a_1$較遠的權重影響變的不明顯,\
也可以再對G做標準化(減掉一個baseline), 但問題是baseline要怎麼求\
因為在訓練過程中, 其實是一邊蒐集資料一邊訓練,\
而且用其他actor(迭代過程)的數據不見得對自己的actor有效\
因此每個actor的action應該只影響自己的actor。\
而且隨機性非常重要, \
一定要讓行為有被採取, 才能知道這個行為的好處,\
所以有時候會希望隨機性大一點。
:::warning
同一個行為對不同actor它的好處是不一樣的\
所以資料更新後, 資料就沒用了(因為狀況不一樣)
:::

#### Critic
Given actor $\theta$, how good it is when observing $s$(and taking action $a$)
簡單來說就是可以觀察現在的狀況跟actor去預測這個episode最後會得多少分。\
Value function : $V^\theta (s)$, When using actor $\theta$, the discounted cumulated reward expects to be obtained after seeing s.\
**Monte-Carlo(MC) based approach**\
用來預測$V^\theta (s)$, 同一個$\theta$看到$s_a$會得到一個接近$G_a'$的答案。, 會看完整個episode。
**Temporal-difference(TD) approach**\
不看完整個episode而是看一小部分就去預測。
\begin{gather}
V^{\theta}(s_t) = r_t + \gamma\ r_{t+1} + \gamma\ r_{t+2} \dots\\
V^{\theta}(s_t) = \gamma\ V^{\theta}(s_{t+1}) + r_t
\end{gather}

但如果有例子是沒看到的就沒辦法處理到。\
而且這裡通常都會預設$\gamma$是1, 這樣就相當於沒有做任何事情。
:::warning
在這裡就可以把前面不知道怎麼算的baseline換成Value Function算出來的值了\
這樣的做法就是把G扣掉很多條可能的路徑的平均。\
但這樣還是不太對
:::

**Advantage Actor-Critic**\
把上面改成Value Function的部分改成平均減平均。
\begin{gather}
\mathop{r_t + V^\theta(s_{t+1})}_{採取a的期望值} - \mathop{V^\theta(s_t)}_{不採取a的期望值}
\end{gather}

#### Reward Shaping
如果做什麼action, reward都是0的話, 沒有辦法判斷action的好壞, train不起來(在只有輸或贏的遊戲的時候)\
因此利用訂定額外的reward去引導機器學習, 需要人類的理解輔助。\
也需要給機器加上好奇心(必須是有意義的), 而且需要解決雜訊(雜訊是沒意義的東西)
:::warning
前面是利用改變機器選擇某些動作的"慾望"\
這裡的分數是直接給機器"reward"或是處罰
:::

#### No-Reward
現實世界定Reward本來就有一些困難, 如果Reward沒有想好可能會有一些很奇怪的行為。\
可以利用人類"示範"來教導機器, 但是要讓機器有能力分辨哪些是"習慣"哪些是要學的\
但即便這樣做, 依舊有可能出現人類不會做的事情(人類不太可能自己開車撞牆\
。

#### Inverse Reinforcement Learning
讓機器自己去訂Reward去學習, 根據人類的行為去訓練Reward Function, 再用Reward Function去訓練RL。\
**Principle : The teacher is always the best**
老師的行為分數一定會高於學生, 但不代表老師都是對的, 不代表要全部都模仿。\
![](https://i.imgur.com/g3oPcs2.jpg)
如圖為大概架構, 老師一定會得到比學生高的分數。

### Chapter 17 : Life-long Learning
我們希望機器不要只能解決一個任務, 而是可以解決多個不同的任務。\
而且隨著進程不斷地重新學習或是學習新的東西, 不要停止訓練。\
或者是同時可以處理不同domain的任務。\
一般來說, 如果我們把兩個任務一次給機器去學習, 機器可能可以訓練出不錯的成果;\
但如果換成先訓練其中一個任務,\
訓練第二個任務時卻會得到機器會"忘記"前面那個任務怎麼做

#### Catastrophic Forgetting(災難性的遺忘)
機器沒辦法分開學會多種任務, 如果有時候運氣好, 資料量比較小,\
可以先把多個任務一起給機器看, 先把Upper bound展示出來。
**Life-long learning v.s. Transfer learning :**\
Transfer learning會著重在第二個任務上, 但life-long learning會希望著重在原本的任務上。
![](https://i.imgur.com/RzBn1ch.jpg)
\begin{aligned}
&Accuracy = \cfrac{1}{T}\sum_{i=1}^T R_{T,i}\\
&Backward\ Transfer = \cfrac{1}{T-1}\sum_{i=1}^{T-1} R_{T,i} - R_{i,i}\\
&(檢查遺忘的程度, 一般的\mbox{life-long learning}沒辦法做到學新任務, 舊任務做更好。)\\
&Forward\ Transfer = \cfrac{1}{T-1}\sum_{i=2}^{T} R_{i-1,i} - R_{0,i}\\
&(還沒看過任務前, 機器學得怎麼樣了)
\end{aligned}

#### Selective Synaptic Plasticity
Regularization-based Approach, 只有部分Neuron可以被修改數值。
\begin{aligned}
&\theta^b\mbox{ is the model learned from the previous tasks, each parameter }\theta^b_i \mbox{ has a "guard" }b_i\\
&L'(\theta) = L(\theta) + \lambda\sum_ib_i(\theta_i - \theta^b_i)^2\\
&L'(\theta) :\mbox{ Loss to be optimized}\\
&L(\theta) :\mbox{ Loss for current task}\\
&b_i : \mbox{How important this parameter is}\\
&\theta_i : \mbox{Parameters to be learning}\\
&\theta^b_i : \mbox{Parameters learned from previous task}\\
\\
&\mbox{if }b_i = 0,\mbox{ there is no constraint on }\theta_i\rightarrow \mbox{Catastrophic Forgetting}\\
&\mbox{if }b_i = \infty,\mbox{ would always be equal to }\theta^b_i\rightarrow \mbox{Intransigence, 可能沒辦法將新任務學好}
\end{aligned}
需要事先去找到哪些參數對$\theta_b$重要, 如果改動參數後結果差距不大則可以改動, 如果差距很大就不要動。\
但限制得太過頭的話, 也會造成新任務學不好。\
不童的$b_i$算法差距也很大。

#### Additional Neural Resource Allocation
**Progressive Neural Networks :** 多開一些Network去處理後面的任務。 每多一個任務, 模型就大一點, 但不會有forget的問題。\
**PackNet** : 一次開大一點的Network, 填入的時候用不同的位置。\
**Compacting, Picking, and Growing(CPG) :** PackNet + Progressive Neural Network

#### Memory Reply
在訓練模型的時候, 順便一起訓練Generator, 當有新的任務的時候可以把generator的輸出當輸入一起丟入訓練。

#### Curriculum Learning
把比較簡單的任務先訓練, 再訓練比較難的(兩個必須有關連性)\
例如 : 手寫辨識先訓練一般的, 再訓練有雜訊的,\
可能在兩個任務上的Performance都可以比較好, \
順序反過來訓練可能就會出現問題(Forgetting)。\
因此, 調換任務學習的順序也是有幫助的。

### Chapter 18 : Network Compression
有時候模型不只會用再Server, 也可能需要用到一些資源有限的環境上(像是手錶或手機), \
這時候我們會需要模型變的小一點, 但不希望他的效果因此而大打折扣。\
一方面也是為了避免隱私問題(上傳資料到Server造成隱私洩漏)。

#### Network Pruning
把Network中一些沒有用的參數剪掉。(也許絕對值越大, 對Network越重要)
![](https://i.imgur.com/vrfJZ3B.jpg)
:::warning
如果一次修剪太多, 可能會無法回復。
:::

**Weight Pruning or Neuron Pruning**\
可以選擇用Weight當單位或是以Neuron當單位, \
如果選擇以Weight修剪, 實作上只會把參數做補0, 但因為沒有刪除任何東西, Network根本不會變小。\
但如果以Neuron作單位, 形狀會變得不規則, 不好實作之外, GPU也不好加速, 因為不規則的時候不容易用矩陣乘法加速運算。
:::warning
用這個方法可以刪除接近95%的參數, Accuracy只掉1%~2%, \
但沒辦法加速實作, 不算是好方法。
:::

**Why Pruning**\
大的Network的參數隨機性比較多, 比較容易Train出好的Model, 挑出好的參數。\
所以只能將大的Network修剪成小的, 沒辦法直接Train小的Network。\
就像大樂透買越多, 中獎機會越高的意思一樣。

#### Knowledge Distillation
把模型大Network訓練出來後,\
將大Network的輸入跟機率分布丟到小Network去Train, 去模仿大Network的輸出。\
這樣的作法,\
因為是模仿大Network的機率分布,\
小Network有機會在還沒看過某個資料前就已經學會如何辨識。\
可以想像成Teacher Network(大Network),\
會提供一些額外的資訊給Student Network(小Network)。
:::warning
可以拿softmax前的任何一層去Train,\
也可以加上更多的限制, 往往加了限制效果會更好。\
也可以在Teacher Network跟Student Network之前加中間Network。
:::

#### Parameter Quantization
1. 只用比較小的參數儲存模型。
2. 參數分群。

可以利用Huffman encoding來讓常用的資源用更少的資源儲存。\
終極型態是希望用一個bits就可以儲存一個參數。(Binary Weights)

#### Architecture Design
![](https://i.imgur.com/PBQnara.jpg)
一般的CNN是有幾個filter就有幾個channel\
**Depthwise Convolution :**\
一個channel就是一個filter, 有幾個channel就有幾個filter。\
這樣的作法, 會使每個channel間沒有任何互動。\
**Pointwise Convolution :**\
filter的大小是1x1, 換句話說就是只考慮channel之間的關聯性。\
通常會跟上面的Depthwise一起用, 可以解決Depthwise的問題。\
**Low rank approximation :**\
中間插入linear layer, 利用矩陣乘法的方式, 可以減少參數量。\
但會減少w的可能性, w會變有限制。

#### Dynamic Computation
![](https://i.imgur.com/LtRfZQB.jpg)
![](https://i.imgur.com/8MOrQVp.jpg)
希望Network可以自由調整運算量, 讓Network判斷問題的難度或是運算資源多寡, 自由去增減Network的深度, 決定要在哪個Layer輸出。


### Chapter 19 : Meta Learning
:::info
Goal : 學習如何學習。
:::
在機器學習這門課中, 最麻煩的一個部分不外乎是調整Hyperparameter,\
沒甚麼方法可以處理這個問題, 只能期待多train之後找出一組最好的。

#### Review ML
1. Function with unknown
2. Define Loss function
3. Optimization


#### Introduction of Meta Learning
我們希望找到一個Learning Algorithm去做學習, 簡單來說就是學習如何學習。\
我們需要學出 : 
1. Network Architecture 網路架構
2. Initial Parameters 初始參數
3. Learning Rate

除了這三個還有很多。

#### Meta Learning - Step 1
不同的Meta Learning的方法就是想去學不同的Component。\
Learnable component($\phi$)就是上面那些Learning Rate那些的東西。\
在開始訓練之前, 需要了解哪些是可以透過學習去學出來的, 哪些不行。

#### Meta Learning - Step 2
Define $\underline{loss\ function}$($L(\phi)$) for $\underline{learning\ algorithm}$\
如果這個algorithm被使用於訓練後,\
對classifier的效果是好的, 代表這個algorithm可能是好的,\
因此如果要評斷algorithm的好壞必須要跑test data。
\begin{gather}
Total\ Loss: L(\phi) = \sum^N_n l^n\\
(\mbox{sum over all the training task})
\end{gather}

:::warning
在Meta Learning的訓練資料, 即一組train set跟test set。
:::

#### Meta Learning - Step 3
\begin{gather}
\phi^* = arg \mathop{min}_\phi L(\phi)
\end{gather}
有可能會遇到沒辦法計算微分的參數, 這時候可以選擇用RL做。

因為meta learning是學習怎麼學習, 只需要一點點有label的資料就好了。

#### What is learnable in a learning algorithm
**Model-Agnostic Meta-Learning :**\
說到底就是訓練怎麼找一個好的初始值去訓練。這個初始值會離正解很近。\
我們可以利用這個方法, 去做多個task, 都會得到不錯的結果, 即便資料的domain不一樣。
**Optimizer :**\
不同的Optimizer對不同的問題適應程度不同, 可以學習找到最好的Optimizer。\
**Network Architecture Search(NAS) :**\
訓練找出最好的Network架構。
\begin{gather}
\hat{\phi} = arg \mathop{min}_\phi L(\phi)\\
\phi : \mbox{Network Architecture}
\end{gather}

一般Network不能微分, 可以利用DARTS硬把Network架構變成可以微分。\
**Data Processing :**\
訓練怎樣的資料前處理可以對訓練效果有幫助。
