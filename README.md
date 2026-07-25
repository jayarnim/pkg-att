# Attention Mechanism Implementation Package

```bash
# INSTALL DEPENDENCIES
conda env create -f env/environment.yaml
conda activate att
```

```py
# LOAD PKG
import att
```

- seminar date: 2024.05.08.

## Attention Score Function

- `dot`: Luong, M. T., Pham, H., & Manning, C. D. (2015, September). Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 conference on empirical methods in natural language processing (pp. 1412-1421).

$$\begin{aligned}
\alpha
&=q^{T}k
\end{aligned}$$

- `scaled`: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

$$\begin{aligned}
\alpha
&=\frac{q^{T}k}{\sqrt{d_{k}}}
\end{aligned}$$

- `prod`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
\alpha
&=h \cdot \mathrm{ReLU}(W \cdot [p \odot q] + b)
\end{aligned}$$

- `cat`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
\alpha
&=h \cdot \mathrm{ReLU}(W \cdot [p \oplus q] + b)
\end{aligned}$$

## simplex projection function

- `smoothed softmax`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
w
&=\frac{\exp{(\alpha)}}{\left[\sum{\exp{(\alpha)}}\right]^{\beta}}
\end{aligned}$$