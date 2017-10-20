---
layout: post
title: "Genetic feature selection"
---

Although [model selection]() plays an important role in learning a signal from some input data, it is arguably even more important to give the algorithm the right input data. The first step for a data scientist, is to construct relevant features by doing appropriate feature engineering. The resulting data set, typically high-dimensional, can then be used as input for a statistical learner.

Although we'd like to think of these learners as smart, and sophisticated, algorithms, they can fall into easy traps. A data scientist has to make the signal as easily identifiable as possible for the model to learn it. In practice, this means that **feature selection** is an important preprocessing step. Feature selections helps to zone in on the relevant variables in a data set, and can also help to eliminate collinear variables. It helps reduce the noise in the data set, and it helps the model pick up the relevant signals.

## Filter and wrapper models

In the above setting, we typically have a high dimensional data matrix $$X \in \mathbb{R}^{n \times p}$$, a target variable $$y$$ (discrete or continuous). A feature selection algorithm will select a subset of $$k << p$$ columns, $$X_S \in \mathbb{R}^{n \times k}$$, that are most relevant to the target variable $$y$$.

Typically, there are two approaches to feature selection:

1. **Wrapper models** use learning algorithms on the original data $$X$$, and selects relevant features based on the (out-of-sample) performance of the learning algorithm.
2. **Filter models** do not use a learning algorithm on the original data $$X$$, but only consider statistical characteristics of the input data.

An example of a wrapper model would be training a random forest on the data $$(X, y)$$, and selecting relevant features based on the feature importances. An example of a filter model would be selecting the features that have the highest correlation with the target $$y$$.

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



## Conclusion



Layout:

**Introduction**:

- Describe DS workflow:
  1. Feature engineering
  2. Feature selection
  3. Model selection
- Focus on feature selection.
- Roughly, there are three types of feature selection approaches (give short examples here)
  1. Filter models
  2. Wrapper models
  3. Hybrid models

**Algorithmic details**

- Zoom in on the filter models, specifically talk about mutual information based approaches.
- Introduce concept of joint mutual information (and why we want to maximise this) + decomposition in relevancy and redundancy. (reference to review paper)

**Optimisation**:

- Lower-dimensional approximation of this criterion and how it results into a family of mutual-information based criteria.
- Genetic algorithms (python example with deap?)
- Variational inference (paper)
- Convex optimisation (paper)

**Discussion**:

- Trade-offs between approaches
- Computation of mutual information terms. (+ link to papers)


References:
(review paper) [1] Brown, Gavin, et al. “Conditional likelihood maximisation: a unifying framework for information theoretic feature selection.” Journal of Machine Learning Research 13.Jan (2012): 27-66.

[2] Quadratic programming feature selection

[3] Variational information maximisation
