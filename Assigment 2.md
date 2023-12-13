# part1
![[Pasted image 20231202175319.png]]

[Explain the code and the test]
# part2
## Exercise 5.1
Compute $f'(x)$.
$$f(x) = log(x^4)sin(x^3)$$
Using the chain rule:
$$f'(x)= \frac{4sin(x^3)}{x} + 3x^2log(x^4)cos(x^3)$$
## Exercise 5.2
Compute the derivative $f′(x)$ of the logistic sigmoid:$$f(x) =\frac{1}{1+exp(-x)}$$Using the chain rule:
$$f'(x) = \frac{\exp(-x)}{(1+\exp(-x))^2}$$
## Exercise 5.3
Compute the derivative $f′(x)$ of the function:
$$
f(x)=\exp (-\frac{1}{2 \sigma^2}(x-\mu)^2)
$$
Using the chain rule:
$$f'(x)= - \frac{x-\mu}{\sigma^2}\exp(-\frac{1}{2 \sigma^2}(x-\mu)^2)$$
## Exercise 5.5
1. For the function $$f_1(x) = \sin(x_1)\cos(x_2) \quad x \in \mathbb{R}^2$$the Jacobian dimension is a $1 \times 2$ matrix (since there's 1 output and 2 inputs). The partial derivatives are $$\frac{\partial f_1}{\partial x_1} = \cos(x_1)\cos(x_2) \quad \text{and} \quad \frac{\partial f_1}{\partial x_2} = -\sin(x_1)\sin(x_2)$$So, the Jacobian is:
$$J_{f_1}(x) = \begin{bmatrix} \cos(x_1)\cos(x_2) & -\sin(x_1)\sin(x_2) \end{bmatrix}$$
2. For the function $$f_2(x,y) = x^⊤y \quad x,y \in \mathbb{R}^n$$the Jacobian dimension is a $1 \times n$ matrix (1 scalar output with  n inputs). The partial derivatives with regards to $x$ are $$\frac{\partial f_2}{\partial x_i} = y_i \quad \forall i \le n$$And, the Jacobian is:
$$J_{f_2}(x,y) = \begin{bmatrix} y_1 & y_2 & \cdots & y_n & x_1  & x_2 & \cdots &x_n\end{bmatrix}$$

3. For the function $$f_3(x) = xx^⊤ \quad x \in \mathbb{R}^n$$the Jacobian dimension is a $n^2 \times n$ matrix. Since $f_3(x)$ outputs a matrix, the Jacobian will not be a 2d matrix but more like a 3-dimensional vector, two dimensions for the output matrix a third dimension for the input vector dimension. 
   The Jacobian is:
$$J_{f_3}(x) =\begin{bmatrix} \frac{\partial(x_1 \cdot x_1)}{\partial x_1} & \frac{\partial(x_1 \cdot x_2)}{\partial x_1} & \dots & \frac{\partial(x_n \cdot x_n)}{\partial x_1} \\\frac{\partial(x_1 \cdot x_1)}{\partial x_2} & \frac{\partial(x_1 \cdot x_2)}{\partial x_2} & \dots & \frac{\partial(x_n \cdot x_n)}{\partial x_2} \\\vdots & \vdots & \ddots & \vdots \\\frac{\partial(x_1 \cdot x_1)}{\partial x_n} & \frac{\partial(x_1 \cdot x_2)}{\partial x_n} & \dots & \frac{\partial(x_n \cdot x_n)}{\partial x_n} \\\end{bmatrix}$$
The partial derivatives are: $$\frac{\partial(x_i \cdot x_i)}{\partial x_i} = 2x_i$$$$\frac{\partial(x_i \cdot x_j)}{\partial x_j} = \begin{cases} x_i & \text{if } i \neq j \\ 0 & \text{if } i = j \end{cases}$$
## Exercise 5.6
Differentiate $f$ with respect to $t$ and $g$ with respect to $X$, where:
$$
\begin{aligned}
& f(\boldsymbol{t})=\sin \left(\log \left(\boldsymbol{t}^{\top} \boldsymbol{t}\right)\right), \quad t \in \mathbb{R}^D \\
& g(\boldsymbol{X})=\operatorname{tr}(\boldsymbol{A} \boldsymbol{X} \boldsymbol{B}), \quad \boldsymbol{A} \in \mathbb{R}^{D \times E}, \boldsymbol{X} \in \mathbb{R}^{E \times F}, \boldsymbol{B} \in \mathbb{R}^{F \times D}
\end{aligned}
$$

1. The derivative of $\sin(x)$ is $\cos(x)$ and the derivative of $\log(x)$ is $\frac{1}{x}$. The derivative of $t^Tt$ with respect to $t$ is $2t$.
$$\frac{\partial f}{\partial t} = \frac{2t\cos(\log(t^Tt))}{t^Tt}$$
2. The derivative with respect to $X$ of $\operatorname{tr}(\boldsymbol{A} \boldsymbol{X} \boldsymbol{B})$ is given by:
$$\frac{\partial g}{\partial X} = A^TB^T$$
   This is because the derivate of the trace function $$\frac{\partial Tr(AXB)}{\partial x_{mj}}=\sum_{i=1}^D\sum_{k=1}^E\sum_{l=1}^Fa_{ik}\frac{\partial x_{kl}}{\partial x_{mj}}b_{li}= \sum_{i=1}^Da_{im}b_{ji}=(BA)_{jm}=(A^TB^T)_{mj}$$
## Exercise 5.7
Compute the derivatives $\frac{\partial f}{\partial x}$of the following functions by using the chain rule. Provide the dimensions of every single partial derivative. Describe your steps in detail.
$$f(z) = log(1 +z) \quad z = x^⊤x, x∈R^D$$
$$f(z) = sin(z) \quad z = Ax+b, A\in R^{E× D},x\in R^D,b\in R^E$$

The derivative of $\log(1 + z)$ with respect to $z$ is $\frac{1}{1 + z}$ and the derivative of $x^T x$ with respect to $x$ is $2x$. Therefore:
$$\frac{\partial f}{\partial x} = \frac{1}{1 + x^T x} * 2x = \frac{2x}{1 + x^T x}$$

The dimension of this derivative is $1 \times D$, with 1 output and D inputs.

The derivative of $\sin(z)$ with respect to $z$ is $\cos(z)$ and the derivative of $Ax + b$ with respect to $x$ is $A$. When we apply a function element-wise to a vector $N\times 1$, the derivative is a diagonal $N\times N$ matrix which has as diagonal the derivative $f'$ applied to $X$ elementwise. Therefore:
$$\frac{\partial f}{\partial x} = diag(\cos(Ax + b)) * A$$
The size of cos(.) is the same as sin(.), or $E\times 1$, hence we would have $(E×1)(E×D)$ that doesn't agree but the matrix $diag(cos(...))$ would have the right dimensions.
The resulting derivative then, is a matrix of dimensions $E \times D$, the same as the dimension of $A$, that is $E \times E$ multiplied by the matrix $A$ of size $E \times D$ which will result in a matrix of dimension $E \times D$.
## Exercise 5.8
Compute the derivatives $\frac{df}{dx}$ of the following functions. Describe your steps in detail. 
1. Use the chain rule. Provide the dimensions of every single partial derivative.
	$f(z) = \exp(-\frac{1}{2}z)$
	$z =g(y) = y^⊤S^{−1}y$
	$y =h(x) =x−\mu$
	where $x,\mu \in R^D, S \in R^{D×D}$

2. $f(x) = tr(xx^⊤ +\sigma^2I), x \in R^D$
   Here tr(A) is the trace of A, i.e., the sum of the diagonal elements $A_{ii}$. Hint: Explicitly write out the outer product.
   
3. Use the chain rule. Provide the dimensions of every single partial derivative. You do not need to compute the product of the partial derivatives explicitly.
   $f =tanh(z) \in R^M$ 
   $z =Ax+b, x\in R^N,A\in R^{M×N},b\in R^M$
   Here, $tanh(\cdot)$ is applied to every component of z.

1. The derivative of $\exp(-\frac{1}{2}z)$ with respect to $z$ is $-\frac{1}{2}\exp(-\frac{1}{2}z)$, the derivative of $y^TS^{-1}y$ with respect to $y$ is $(S^{-1} + S^{-T}) y$, and the derivative of $x - \mu$ with respect to $x$ is 1.

   Applying the chain rule, we get:

   $\frac{\partial f}{\partial x} = -\frac{1}{2}\exp(-\frac{1}{2}y^TS^{-1}y) * (S^{-1} + S^{-T})(x - \mu)$

   The dimension of this derivative is $1 \times D$, the same as the dimension of $x$.

2. For the function $f(x) = \text{tr}(xx^T + \sigma^2I)$ where $x \in \mathbb{R}^D$, the derivative with respect to $x$ is given by:

   $\frac{\partial f}{\partial x} = 2x$

	The dimension of this derivative is $1 \times D$.

3. For the function $f = \tanh(z) \in \mathbb{R}^M$ where $z = Ax + b$, $x \in \mathbb{R}^N$, $A \in \mathbb{R}^{M \times N}$, and $b \in \mathbb{R}^M$, we again use the chain rule to compute the derivative.

   The derivative of $\tanh(z)$ with respect to $z$ is $1 - \tanh^2(z)$ and the derivative of $Ax + b$ with respect to $x$ is $A$.

   Applying the chain rule, we get:

   $\frac{\partial f}{\partial x} = diag(1 - \tanh^2(Ax + b)) * A$

   Here, the $1 - \tanh^2(\cdot)$ function is applied element-wise to the vector $Ax + b$. The resulting derivative is a matrix of dimensions $M \times N$, the same as the dimension of $A$, with the same reasoning of an above exercise.

## Exercise 5.9
We define 
$$g(x,z,\nu) := \log p(x,z) − \log q(z,\nu)$$
$$z := t(\epsilon,\nu)$$
for differentiable functions $p,q,t$ and $x \in R^D,z \in R^E,\nu \in R^F,\epsilon \in R^G$. By using the chain rule, compute the gradient:
$$\frac{d}{d\nu}g(x,z,\nu)$$

The derivative of $\log p(x, z)$ with respect to $z$ is $\frac{1}{p(x,z)} \frac{\partial p(x,z)}{\partial z}$
Then, the derivative of $-\log q(z, \nu)$ with respect to $z$ is $-\frac{1}{q(z, \nu)} \frac{\partial q(z, \nu)}{\partial z}$ and the derivative of $z = t(\epsilon, \nu)$ with respect to $\nu$ is $\frac{\partial t(\epsilon, \nu)}{\partial \nu}$.

Applying the chain rule, we get:

$\frac{d}{d\nu}g(x, z, \nu) = \frac{1}{p(x,z)} \frac{\partial p(x,z)}{\partial z}\frac{\partial t(\epsilon, \nu)}{\partial \nu} -\frac{1}{q(z, \nu)} \frac{\partial q(z, \nu)}{\partial z} \frac{\partial t(\epsilon, \nu)}{\partial \nu}$