# nuFFT

nuFFT:
$$
\hat{f}(t_m) = \hat{f}_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} \exp\left(\frac{-2\pi i}{N}\left(n-c^N\right)t_m\right)f_n
$$

nuFFT adjoint:
$$
f_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{M-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m\right)\hat{f}_m
$$

$\mathrm{kPSF}(t)_{l}^{N}$:
$$
\mathrm{kPSF}_l^N = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} \exp\left(\frac{-2\pi i}{N}(l-c^N)(n-c^N)\right)\frac{1}{\sqrt{N}}\sum_{m=0}^{M-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m\right)k_m
$$
with a kernel $k_m$ and $k_m=1$ in the simplest case.

Toeplitz embedding requires $\mathrm{kPSF}(2t)_{l}^{2N}$.
Even (2l) and odd (2l+1) indices of $\mathrm{kPSF}(2t)_{l}^{2N}$ can be computed efficiently on a grid of size $N$, i.e.
$$
\mathrm{kPSF}(2t)_{2l+s}^{2N}= \frac{1}{\sqrt{N}}\underbrace{\sum_{n=0}^{N-1} \exp\left(-\frac{2\pi i}{N}nl\right)}_{\mathrm{FFT}^N}\underbrace{\frac{1}{\sqrt{N}}\sum_{m=0}^{M-1}\exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m'\right)}_{\mathrm{nuFFT}^N}\cos\left(\pi t'_m\right)\underbrace{\exp\left(2\pi i\left(\frac{c^N}{N}-\frac{1}{2}\right)t'_m\right)}_{=1\text{ if N even}}k_m
$$

$$
\begin{aligned}
\mathrm{kPSF}(2t)_{2l+s}^{2N}
&= \frac{1}{\sqrt{2N}}\sum_{n=0}^{2N-1} \exp\left(\frac{-2\pi i}{2N}(2l+s-c^{2N})(n-c^{2N})\right)\frac{1}{\sqrt{2N}}\sum_{m=0}^{M-1} \exp\left(\frac{2\pi i}{2N}\left(n-c^{2N}\right)2t_m\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{2N-1} \exp\left(\frac{2\pi i}{2N}\left(n-c^{2N}\right)\left(2t_m-2l-s+c^{2N}\right)\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{2N}\left(n-c^{2N}\right)\left(2t_m-2l-s+c^{2N}\right)\right)\left(1 + \exp\left(\frac{2\pi i}{2N}N\left(2t_m-2l-s+c^{2N}\right)\right)\right)k_m\\
&\overset{t'_m=t_m+\frac{c^{2N}-s}{2}}= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{2N}\left(n-c^{2N}\right)\left(2t_m'-2l\right)\right)\left(1 + \exp\left(\frac{2\pi i}{2N}N\left(2t'_m-2l\right)\right)\right)k_m\\
&\overset{c^{2N}=N}= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^{2N}\right)\left(t_m'-l\right)\right)\left(1 + \exp\left(2\pi i t'_m\right)\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)\left(t_m'-l\right)\right)\exp\left(\frac{2\pi i}{N}\left(c^N-c^{2N}\right)\left(t_m'-l\right)\right)\left(1 + \exp\left(2\pi i t'_m\right)\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)\left(t_m'-l\right)\right)\exp\left(\frac{2\pi i}{N}\left(c^N-c^{2N}\right)\left(-l\right)\right)\exp\left(\frac{2\pi i}{N}\left(c^N-c^{2N}\right)\left(t_m'\right)\right)\left(1 + \exp\left(2\pi i t'_m\right)\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)\left(t_m'-l\right)\right)\exp\left(-\frac{2\pi i}{N}c^Nl\right)\exp\left(\frac{2\pi i}{N}\left(c^N-c^{2N}\right)\left(t_m'\right)\right)\left(1 + \exp\left(2\pi i t'_m\right)\right)k_m\\
&= \frac{1}{2N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)\left(t_m'-l\right)\right)\exp\left(-\frac{2\pi i}{N}c^Nl\right)\exp\left(2\pi i\frac{c^N}{N}\left(t_m'\right)\right)\left(1 + \exp\left(-2\pi i t'_m\right)\right)k_m\\
&= \frac{1}{N}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \exp\left(\frac{2\pi i}{N}\left(n-c^N\right)\left(t_m'-l\right)\right)\exp\left(-\frac{2\pi i}{N}c^Nl\right)\frac{\exp\left(2\pi i\frac{c^N}{N}\left(t_m'\right)\right) + \exp\left(2\pi i \left(\frac{c^N}{N}-1\right)t'_m\right)}{2}k_m\\
&= \frac{1}{N}\sum_{n=0}^{N-1} \exp\left(-\frac{2\pi i}{N}\left(n-c^N\right)l\right)\exp\left(-\frac{2\pi i}{N}c^Nl\right)\sum_{m=0}^{M-1}\exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m'\right)\frac{\exp\left(2\pi i\frac{c^N}{N}\left(t_m'\right)\right) + \exp\left(2\pi i \left(\frac{c^N}{N}-1\right)t'_m\right)}{2}k_m\\
&= \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} \exp\left(-\frac{2\pi i}{N}nl\right)\frac{1}{\sqrt{N}}\sum_{m=0}^{M-1}\exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m'\right)\frac{\exp\left(2\pi i\frac{c^N}{N}\left(t_m'\right)\right) + \exp\left(2\pi i \left(\frac{c^N}{N}-1\right)t'_m\right)}{2}k_m\\
&= \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} \exp\left(-\frac{2\pi i}{N}nl\right)\frac{1}{\sqrt{N}}\sum_{m=0}^{M-1}\exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m'\right)\frac{\exp\left(\pi i t_m'\right) + \exp\left(-\pi i t'_m\right)}{2}\exp\left(2\pi i\left(\frac{c^N}{N}-\frac{1}{2}\right)t'_m\right)k_m\\
&= \frac{1}{\sqrt{N}}\underbrace{\sum_{n=0}^{N-1} \exp\left(-\frac{2\pi i}{N}nl\right)}_{\mathrm{FFT}^N}\underbrace{\frac{1}{\sqrt{N}}\sum_{m=0}^{M-1}\exp\left(\frac{2\pi i}{N}\left(n-c^N\right)t_m'\right)}_{\mathrm{nuFFT}^N}\cos\left(\pi t'_m\right)\underbrace{\exp\left(2\pi i\left(\frac{c^N}{N}-\frac{1}{2}\right)t'_m\right)}_{=1\text{ if N even}}k_m\\
\end{aligned}
$$
