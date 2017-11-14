---
layout: post
title: "Mutual information based feature selection"
---

Although [model selection](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/) plays an important role in learning a signal from some input data, it is arguably even more important to give the algorithm the right input data. The first step for a data scientist, is to construct relevant features by doing appropriate feature engineering. The resulting data set, typically high-dimensional, can then be used as input for a statistical learner.

Although we'd like to think of these learners as smart, and sophisticated, algorithms, they can be fooled by all the weird correlations present in your data. A data scientist has to make the signal as easily identifiable as possible for the model to learn it. In practice, this means that **feature selection** is an important preprocessing step. Feature selection helps to zone in on the relevant variables in a data set, and can also help to eliminate collinear variables. It helps reduce the noise in the data set, and it helps the model pick up the relevant signals.

## Filter methods

In the above setting, we typically have a high dimensional data matrix $$X \in \mathbb{R}^{n \times p}$$, and a target variable $$y$$ (discrete or continuous). A feature selection algorithm will select a subset of $$k << p$$ columns, $$X_S \in \mathbb{R}^{n \times k}$$, that are most relevant to the target variable $$y$$.

In general, we can divide feature selection algorithms as belonging to one of three classes:

1. **Wrapper methods** use learning algorithms on the original data $$X$$, and selects relevant features based on the (out-of-sample) performance of the learning algorithm. Training a random forest on the data $$(X, y)$$, and selecting relevant features based on the feature importances would be an example of a wrapper model.
2. **Filter methods** do not use a learning algorithm on the original data $$X$$, but only consider statistical characteristics of the input data. For example, we can select the features for which the correlation between the feature and the target variable exceeds a correlation threshold.
3. **Embedded methods** are a catch-all group of techniques which perform feature selection as part of the model construction process. The LASSO is an example of an embedded method.

In this blog post I will focus on filter methods, and in particular I'll look at filter methods that use an entropy measure called **mutual information** to assess which features should be included in the reduced data set $$X_S$$. The resulting criterion results in an NP-hard optimisation problem, and I'll discuss several ways in which we can try to find optimal solutions to the problem.

<!--excerpt-->

## Joint mutual information

Mutual information is a measure between two (possible multi-dimensional) random variables $$X$$ and $$Y$$, that quantifies the amount of information obtained about one random variable, through the other random variable. The mutual information is given by

$$ I(X; Y) = \int_X \int_Y p(x, y) \log \frac{p(x, y)}{p(x) p(y)} dx dy, $$

where $$p(x, y)$$ is the joint probability density function of $$X$$ and $$Y$$, and where $$p(x)$$ and $$p(y)$$ are the marginal density functions. The mutual information determines how similar the joint distribution $$p(x, y)$$ is to the products of the factored marginal distributions.

When it comes to feature selection, we would like to maximise the
**joint mutual information** between the subset of selected features $$X_S$$, and the target $$y$$

$$ \tilde{S} = \arg \max_S I(X_S; y), \quad\quad s.t. |S| = k, $$

where $$k$$ is the number of features we want to select. This is an NP-hard optimisation problem, because the set of possible combinations of features grows exponentially.

## Greedy solution to the feature selection problem

The simplest approach to solve this optimisation problem, is by using a greedy forward step-wise selection algorithm. Here, features are selected incrementally, one feature at a time.

Let $$S^{t - 1} = \{x_{f_1}, \ldots, x_{f_{t - 1}}\}$$ be the set of selected features at time step $$t - 1$$. The greedy method selects the next feature $$f_t$$ such that

$$ f_t = \arg\max_{i \notin S^{t - 1}} I(X_{S^{t - 1} \cup i} ; y) $$

Greedy feature selection thus selects the features that at each step results in the biggest increase in the joint mutual information. Computing the joint mutual information involves integrating over a high-dimensional space, which quickly becomes intractable computationally. To make this computation a bit easier, we make the following assumption on the data:

* **Assumption 1**: The selected features $$X_S$$ are independent and class-conditionally independent given the unselected feature $$X_k$$ under consideration.

Under this assumption one can show, by decomposing the mutual information term, that solving the above problem is the same as solving

$$ f_t = \arg\max_{i \notin S^{t - 1}} \underbrace{I(x_i; y)}_{\text{relevancy}} - \underbrace{\left[I(x_i; x_{S^{t - 1}}) - I(x_i; x_{S^{t - 1}} | y) \right]}_{\text{redundancy}}.$$

Optimising this criterion results in trading off the *relevance* of a new feature $$x_i$$ with respect to the target $$y$$, against the *redundancy* of that information compared to the information contained in the variables $$X_{S^{t - 1}}$$ that are already selected.

## Lower-dimensional approximation

Even with the above simplification of the joint mutual information, the quantities involving $$S^{t - 1}$$ are still $$(t - 1)$$-dimensional integrals. By making some assumptions on the data, we can simplify the computation of the mutual information terms drastically.

* **Assumption 2**: All features are pairwise class-conditionally independent, i.e.
$$
p(x_i x_j | y) = p(x_i | y)p(x_j | y)
$$
  This implies that $$\sum I(X_j; X_k | y)$$ will be zero.
* **Assumption 3**: All features are pairwise independent, i.e.
$$
p(x_i x_j) = p(x_i) p(x_j)
$$
  This implies that $$\sum I(X_j; X_k)$$ will be zero.

To make the problem tractable, almost all approaches in the literature use the above assumptions to propose the following low-order approximations

$$
\begin{align}
I(x_i ; x_{S^{t - 1}}) \approx \alpha \sum_{k = 1}^{t-1} I(x_{f_k}; x_i), \\
I(x_i ; x_{S^{t - 1}} | y) \approx \beta \sum_{k = 1}^{t - 1} I(x_{f_k}; x_i | y).
\end{align}
$$

Hence, the optimization problem now simplifies to

$$
f_t = \arg\max_{i \notin S^{t - 1}} \underbrace{I(x_i; y)}_{\text{relevancy}} - \underbrace{\left[ \alpha \sum_{k = 1}^{t-1} I(x_{f_k}; x_i) - \beta \sum_{k = 1}^{t - 1} I(x_{f_k}; x_i | y) \right]}_{\text{redundancy}}
$$

where $$\alpha$$ and $$\beta$$ are to be specified.

## A family of feature selection algorithms

These parameters actually specify a family of mutual information-based criteria, and we can recover some prominent examples for specific values of $$\alpha$$ and $$\beta$$:

* Joint mutual information (JMI): $$\alpha = \frac{1}{t - 1}$$ and $$\beta = \frac{1}{t - 1}$$.
* Maximum relevancy minimum redundancy (MRMR): $$\alpha = \frac{1}{t - 1}$$ and $$\beta = 0$$.
* Mutual information maximisation (MIM): $$\alpha = 0$$ and $$\beta = 0$$.

Each of these criteria make different assumptions, and we can see that $$\alpha$$ and $$\beta$$ control the degree of belief in one of the assumptions.

* A value of $$\alpha$$ closer to zero indicates a stronger belief in Assumption 3.
* A value of $$\beta$$ closer to zero indicates a stronger belief in Assumption 2.
* All variable selection criteria make Assumption 1.

Brown et al (2012)[^1] perform a number of experiments in which they compare the different algorithms against each other. They find that algorithms that balance the relative magnitude of relevancy against redundancy, tend to perform well in terms of stability and the accuracy of the final learning algorithm. They suggest that the JMI, and MRMR, criteria should be the go-to mutual information based criteria to consider for feature selection.

In practice, your results will not only depend on the criterion used. Even though we have reduced the computation of the full joint mutual information to pairwise (conditional) mutual information terms, their computation is still non-trivial and forms an area of active research. An interesting recent contribution has been made by Gao et al (2017)[^2], where they propose an estimator that handles the case where $$X$$ and $$Y$$ can both be a mixture of a continuous and discrete distribution. More estimators of $$I(X; Y)$$ exist in the literature, and the accuracy of the approximation you use will influence the results of your feature selection algorithm.

## Practical advice

No single method will always work out of the box on any new problem. So how can we decide on the type of feature selection algorithm, a filter or embedded method, to use? Some things to take into consideration in practice are:

* **Low sample size**: If you're dealing with a data set that has a low sample size (<1000s), be mindful that the computation of the mutual information may break down. There are different ways in which one can compute the mutual information, so make sure to check what approximation your implementation uses, and check the relevant papers to see what their performance is like in low sample size regimes.

* **High dimensionality and sparse relationship between features and target**: In high dimensional, and sparse settings, random forest based feature selection algorithms may have trouble identifying the relevant features due to the random subspace component of the learning algorithm. In this case it is good to check stability of the algorithm on bootstrapped samples of the original data.

* **Low sample size and high dimensional space**: This is one of the hardest settings to work in. Typically, an algorithm called stability selection[^3] with a LASSO structure learner works well here. Stability selection is a very conservative method however, and will only select variables that have a relatively strong relationship with the target variable.

Regardless of the above advice, if you have sufficient data it is always a smart idea to try multiple feature selection algorithms on your data set, and to compare the results. It is helpful to compare both the performance of your final learner on a separate hold-out set, where feature selection is done only on the training set, and the stability of the feature selection algorithms. The stability of your algorithm can for example be assessed by using Kunchecha's stability index[^4], or Yu et al's stability index[^5].

## References

[^1]: Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). Conditional likelihood maximisation: a unifying framework for information theoretic feature selection. Journal of machine learning research, 13(Jan), 27-66. http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

[^2]: Gao, W., Kannan, S., Oh, S., & Viswanath, P. (2017). Estimating mutual information for discrete-continuous mixtures. arXiv preprint arXiv:1709.06212.. https://arxiv.org/pdf/1709.06212.pdf

[^3]: Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473. https://stat.ethz.ch/~nicolai/stability.pdf

[^4]: Kuncheva, L. I. (2007, February). A stability index for feature selection. In Artificial intelligence and applications (pp. 421-427). http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.6458&rep=rep1&type=pdf

[^5]: Yu, L., Ding, C., & Loscalzo, S. (2008, August). Stable feature selection via dense feature groups. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 803-811). ACM. https://pdfs.semanticscholar.org/45e2/ee33164d6fac44178196e09733b7628814e2.pdf
