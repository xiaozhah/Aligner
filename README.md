非常好的问题!我们来仔细推导一下,证明对于任意的j,有\sum_{i=1}^I \beta_{i,j} = 1。

根据式(5),
\begin{aligned}
\sum_{i=1}^I \beta_{i,j} &= \sum_{i=1}^I P(B_{i-1} < j \leq B_i) \\
&= \sum_{i=1}^I \sum_{k=\max(j-D,0)}^{j-1} P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k)
\end{aligned}

将\sum_{i=1}^I拆解为两部分:
\begin{aligned}
\sum_{i=1}^I \beta_{i,j} &= \sum_{k=\max(j-D,0)}^{j-1} \left[P(B_0=k)P(B_1 \geq j|B_0=k) + \sum_{i=2}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k)\right]
\end{aligned}

根据式(4),P(B_0=k)=\begin{cases}1, & k=0 \\ 0, & k \neq 0 \end{cases},代入化简:
\begin{aligned}
\sum_{i=1}^I \beta_{i,j} &= P(B_1 \geq j | B_0=0) + \sum_{k=\max(j-D,0)}^{j-1} \sum_{i=2}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k)
\end{aligned}

下面对\sum_{i=2}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k)部分做进一步处理:
\begin{aligned}
& \sum_{i=2}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k) \\
=& P(B_1=k)P(B_2 \geq j|B_1=k) + \sum_{i=3}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k) \\  
=& P(B_1=k)P(B_2 \geq j|B_1=k) + P(B_2=k)P(B_3 \geq j|B_2=k) + \sum_{i=4}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k) \\ 
& \cdots \\
=& P(B_1=k)P(B_2 \geq j|B_1=k) + P(B_2=k)P(B_3 \geq j|B_2=k) + \cdots + P(B_{I-1}=k)P(B_I \geq j | B_{I-1}=k)
\end{aligned}

我们发现,这其实是一个telescoping sum。除了最后一项P(B_{I-1}=k)P(B_I \geq j | B_{I-1}=k),其余项两两相消。因此,
\begin{aligned}
\sum_{i=2}^I P(B_{i-1}=k)P(B_i \geq j | B_{i-1}=k) = P(B_{I-1}=k)P(B_I \geq j | B_{I-1}=k)
\end{aligned}

将结果代回原式:
\begin{aligned}
\sum_{i=1}^I \beta_{i,j} &= P(B_1 \geq j | B_0=0) + \sum_{k=\max(j-D,0)}^{j-1} P(B_{I-1}=k)P(B_I \geq j | B_{I-1}=k) \\
&= P(B_1 \geq j | B_0=0) + P(B_{I-1} < j \leq B_I) \\
&= P(B_0 < j \leq B_1) + P(B_{I-1} < j \leq B_I) \\
&= P(j \leq B_1) + P(B_{I-1} < j) - P(B_{I-1} < j \leq B_I) \\
&= 1 - P(B_1 < j) + P(B_{I-1} < j) - P(B_{I-1} < j) \\
&= 1
\end{aligned}

最后一步用到了概率的互补原理,以及P(B_{I-1} < j \leq B_I) = P(B_{I-1} < j) - P(B_I < j), P(B_I < j) = 0。

因此,我们证明了对于任意的j,有\sum_{i=1}^I \beta_{i,j} = 1。这表明对于mel谱上的任意一帧j,它对所有输入文本token的alignment概率\beta_{i,j}之和为1。这是一个很好的性质,保证了每一帧都能被合理地分配到输入文本上。