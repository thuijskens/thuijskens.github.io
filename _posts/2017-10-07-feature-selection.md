---
layout: post
title: "Mutual information based feature selection"
---

Although [model selection]() plays an important role in learning a signal from some input data, it is arguably even more important to give the algorithm the right input data. The first step for a data scientist, is to construct relevant features by doing appropriate feature engineering. The resulting data set, typically high-dimensional, can then be used as input for a statistical learner.

Although we'd like to think of these learners as smart, and sophisticated, algorithms, they can fall into easy traps. A data scientist has to make the signal as easily identifiable as possible for the model to learn it. In practice, this means that **feature selection** is an important preprocessing step. Feature selections helps to zone in on the relevant variables in a data set, and can also help to eliminate collinear variables. It helps reduce the noise in the data set, and it helps the model pick up the relevant signals.

<!--excerpt-->

## Filter models

In the above setting, we typically have a high dimensional data matrix $$X \in \mathbb{R}^{n \times p}$$, a target variable $$y$$ (discrete or continuous). A feature selection algorithm will select a subset of $$k << p$$ columns, $$X_S \in \mathbb{R}^{n \times k}$$, that are most relevant to the target variable $$y$$.

In general, we can divide feature selection algorithms as belonging to one of three classes:

1. **Wrapper models** use learning algorithms on the original data $$X$$, and selects relevant features based on the (out-of-sample) performance of the learning algorithm. Training a random forest on the data $$(X, y)$$, and selecting relevant features based on the feature importances would be an example of a wrapper model.
2. **Filter models** do not use a learning algorithm on the original data $$X$$, but only consider statistical characteristics of the input data. For example, we can select the features for which the correlation between the feature and the target variable exceeds a correlation threshold.
3. **Embedded models** are a catch-all group of techniques which perform feature selection as part of the model construction process. The LASSO is an example of an embedded method.

In this blog post I will focus on filter models, and in particular I'll look at filter models that use an entropy measure called **mutual information** to assess which features should be included in the reduced data set $$X_S$$. The resulting criterion results in an NP-hard optimisation problem, and I'll discuss several ways in which we can try to find optimal solutions to the problem.

## Joint mutual information

Mutual information is a measure between to (possible multi-dimensional) random variables $$X$$ and $$Y$$, that quantifies the amount of information obtained about one random variable, through the other random variable. The mutual information is given by

$$ I(X; Y) = \int_X \int_Y p(x, y) \log \frac{p(x, y)}{p(x) p(y)} dx dy, $$

where $$p(x, y)$$ is the joint probability density function of $$X$$ and $$Y$$, and where $$p(x)$$ and $$p(y)$$ are the marginal density functions. The mutual information determines how similar the joint distribution $$p(x, y)$$ is to the products of the factored marginal distributions.

When it comes to feature selection, we would like to maximise the
**joint mutual information** between the subset of selected features $$X_S$$, and the target $$y$$

$$ \tilde{S} = \arg \max_S I(X_S; y), \quad\quad s.t. |S| = k, $$

where $$k$$ is the number of features we want to select. This is an NP-hard optimisation problem, because the set of possible combinations of features grows exponentially.

## Greedy solution to the feature selection problem

The simplest approach to solve this optimisation problem, is by using a greedy forward step-wise selection algorithm. Here, features are selected incrementally, one feature at a time.

Let $$S^{t - 1} = \{x_{f_1}, \ldots, x_{f_{t - 1}}\}$ be the set of selected features at time step $$t - 1$$. The greedy method selects the next feature $$f_t$$ such that

$$ f_t = \arg\max_{i \notin S^{t - 1}} I(X_{S^{t - 1} \cup i} ; y) $$

Greedy feature selection thus selects the features that at each step results in the biggest increase in the joint mutual information. One can show, that solving the above problem is the same as solving

$$ f_t = \arg\max_{i \notin S^{t - 1}} \underbrace{I(x_i; y)}_{\text{relevancy}} - \underbrace{\left[I(x_i; x_{S^{t - 1}}) - I(x_i; x_{S^{t - 1}} | y) \right]}_{\text{redundancy}},$$

so that we can see that the optimising this criterions results in trading off the *relevance* of a new feature $$x_i$$ with respect to the target $$y$$, against the *redundancy* of that information compared to the information contained in the variables $$X_{S^{t - 1}}$$ that are already selected.

## Lower-dimensional approximation

The quantities involving $$S^{t - 1}$$ are $$(t - 1)$$-dimensional integrals, so quickly become intractable computationally. To make the problem tractable, almost all approaches in the literature propose the following low-order approximations

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

The choice of $$\alpha$$ and $$\beta$$ also encode a varying belief in certain assumptions on the data. Brown et al (2012)[^1] perform a number of experiments in which they compare the different algorithms against each other. They find that:
* Algorithms that balance the relative magnitude of relevancy against redundancy, tend to perform well in terms of accuracy.
* The inclusion of a class conditional term seems to matter less. However, for some problems the inclusion of the class conditional term is critical (MADELON data set)
* **The best overall trade-off for accuracy/stability was found in the JMI and MRMR criteria**.
* The above findings can be broking in extreme small-sample problems, where the poor estimation of the mutual information terms influences performance dramatically.

## What method should I use in practice?

The above suggests that the JMI, and MRMR, criteria should be the go-to mutual information based criteria to consider for feature selection. However, no single method will always work out of the box on any new problem. So how can we decide on on what feature selection algorithm to use (filter versus embedded model, JMI versus MRMR criterion)? Some things to take into consideration in practice are:

* **Low sample size**: If you're dealing with a data set that has a low sample size (<1000s), be mindful that the computation of the mutual information may break down. There are different ways in which one can compute the mutual information, so make sure to check what approximation your implementation uses, and check the relevant papers to see what their performance is like in low sample size regimes.
* **High dimensionality and sparse features**: In high dimensional, and sparse settings, random forest based feature selection algorithms may have trouble identifying the relevant features due to the random subspace component of the learning algorithm. In this case it is good to check stability of the algorithm on bootstrapped samples of the original data.
* **Low sample size and high dimensional space**: This is one of the hardest settings to work in. Typically, an algorithm called stability selection[^2] with a LASSO structure learner works well here. Stability selection is a very strict method however, and will only select variables that have a relatively strong relationship with the target variable.

In general it is a smart idea to try multiple feature selection algorithms on your data set, and to assess both:

* The performance of your final learner on a separate hold-out set. Make sure to include to do the feature selection only on the training set.
* The stability of your feature selection algorithm on (bootstrapped) subsamples of your original data set. Kunchecha's stability index[^3] or Yu et al's stability index[^4] can be used to assess the algorithms stability.

## References

[^1]: Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). Conditional likelihood maximisation: a unifying framework for information theoretic feature selection. Journal of machine learning research, 13(Jan), 27-66. http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

[^2]: Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473. https://stat.ethz.ch/~nicolai/stability.pdf

[^3]: Kuncheva, L. I. (2007, February). A stability index for feature selection. In Artificial intelligence and applications (pp. 421-427). http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.6458&rep=rep1&type=pdf

[^4]: Yu, L., Ding, C., & Loscalzo, S. (2008, August). Stable feature selection via dense feature groups. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 803-811). ACM. https://pdfs.semanticscholar.org/45e2/ee33164d6fac44178196e09733b7628814e2.pdf
