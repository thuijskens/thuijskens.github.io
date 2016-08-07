---
layout: post
title: Causal analysis
---

*"Correlation does not imply causation"* is one of the mantras every person that works with data should know. It is one of the first concepts taught in any intro to statistics class. There is a good reason for this, as most of the work of a data scientist or a statistician does actually revolves around questions of causation:

* Did customers buy into product X or service Y because of last weeks email campaign, or would they have converted regardless of whether we did or did not run the campaign?
* Was there any effect of in-store promotion Z on the spending behavior of customers four weeks post the promotion?
* Did people with disease X got better because they took treatment Y, or would they have gotten better anyways?

Being able to distinguish between spurious correlations and true causal effects means a data scientist can identify which truly add value to the company.

This is where traditional statistics, like experimental design, comes into play. Although it is perhaps not commonly associated with the field of data science, more and more data scientists are using principles from experimental design. [Data scientists at Twitter](https://www.meetup.com/Data-Science-London/events/226696208/) use these principles to correct for hidden bias in their A/B tests, engineers from Google have developed a whole [R package](https://google.github.io/CausalImpact/) around causal analysis, and at Tesco we use these principles to attribute changes in customer spending behavior to promotions customers participated in.

In this post we will have a look at some of the frequently used methods in causal analysis. First we will go through a little bit of theory and talk about why we need causal analysis in the first place. Finally, we'll actually code up a propensity score matching algorithm in Python and go through a real example.

### The fundamental problem of causal analysis

Before we delve deeper into some of these principles, let's define some standard terminology. We will consider a binary treatment variable $$T \in \{0, 1\}$$, where $$T = 1$$ denotes that one received the treatment and $$T = 0$$ denotes otherwise. $$T$$ could be any phenomenon we want to measure the causal effect of. It could be a new drug for a disease, a new kind of supplement, a new website design or a commercial campaign you ran.

We also define a variable $$Y_T$$, that denotes the *response* for each value of the treatment $$T$$. For example, in case $$T$$ was a promotion we ran in-store, $$Y_1$$ would be the response (which could be something like customer spend) if a customer participated in the promotion and $$Y_0$$ would be the response otherwise. I am suppressing any subscripts $$i$$ to identify individual observations for convenience here.

Finally, we introduce another variable $$Y$$, that denotes the actual *observed* value of the response. Usually, we are interested in either the *average treatment effect* (ATE)

$$ ATE = \mathbb{E}\left[\delta\right] = \mathbb{E}\left[Y_1 - Y_0\right],$$

which is the average (over the whole population) of the individual level causal effect $$\delta$$, or in the *average treatment effect on the treated* (ATT)

$$ ATT = \mathbb{E}\left[\delta \,\vert\, T = 1\right] = \mathbb{E}\left[Y_1 - Y_0 \,\vert\, T = 1\right],$$

which is the average of the individual level causal effect for the people that got treated. In case of our promotion example, the ATE would the average difference in customer spend between customers that did and customers that did not participate in the promotion, and the ATT would be the average effect only for customers that participated.

However, we will run into a problem when we try to directly compute these quantities. For a customer that participated in the promotion, we can never measure **both** the response in case they did participate and the response in case they did not participate simultaneously. We never observe both $$Y_1$$ and $$Y_0$$ for the same individual. The fact that we are missing either $$Y_1$$ or $$Y_0$$ for every observation is sometimes referred to as the *fundamental problem of causal analysis*.

### Randomized trials as the gold standard

How can we solve this problem? Let's focus on estimating the ATE for now. We would like to compute this effect directly, but in reality we only have the estimator

$$\hat{ATE} = \mathbb{E}\left[Y_1 \,\vert\, T = 1\right] - \mathbb{E}\left[Y_0 \,\vert\, T = 0\right].$$

The ATE measures *causation*, but the above estimator only measures *association*. In general, these two are not equal and this is where the phrase "correlation does not imply causation" comes from. It is also referred to as the *identification problem*: we can't easily separate causation from association.

To tackle this problem, we will need to make some assumptions, as assumption-free causal analysis is impossible. One can show that under the assumptions

* $$\mathbb{E}\left[Y_1 \,\vert\, T = 1\right] = \mathbb{E}\left[Y_1 \,\vert\, T = 0\right] = \mathbb{E}\left[Y_1\right],$$ and;
* $$\mathbb{E}\left[Y_0 \,\vert\, T = 1\right] = \mathbb{E}\left[Y_0 \,\vert\, T = 0\right] = \mathbb{E}\left[Y_0\right]$$,

the estimator $$\hat{ATE}$$ is a consistent and unbiased estimator of the ATT, because

$$ \mathbb{E}\left[Y_1 \,\vert\, T = 1\right] - \mathbb{E}\left[Y_0 \,\vert\, T = 0\right] = \mathbb{E}\left[Y_1\right] - \mathbb{E}\left[Y_0\right],$$

which means that *association* actually equals *causation* under these assumptions.

In words, these two assumptions mean that, *on average*, people in the test group and people in the control group are similar to each other, with respect to the potential outcomes. It means that for each group we should only compare people that are similar to each other (comparing apples to apples).

This can be achieved by using a random sampling design, where you randomly assign treatment to observations in your population, and this is why a randomized experiment is sometimes called the *gold standard* in experimental design.

### Observational studies

In reality, we mostly deal with data that does not come from a carefully set up randomized experiment. We deal with observational studies where treatment assignment is typically not random and influenced by external factors. In case of the promotion example, we have no influence in who does or does not participate in the promotion and participation is typically driven by external factors, like how much money people can spend.

How do we still come up with unbiased estimates of the ATE and ATT? One popular approach to this problem is to use matching methods.

### Matching methods

 Matching methods try to structure the observational data in such a way, that we could think of the data *as if* it was generated by a random sampling design. To understand the idea behind these methods, let's have a look at the promotion example.

Imagine that there is a customer called Bob, and that Bob has an identical twin-brother called Bill. Bob and Bill have the same shopping habits, they like the same products and they have the same amount of money to spent and they prefer to do one big weekly shop instead of shopping everyday at their grocery store.

Their grocery store now launches a big promotion promotion their new line of products, and Bob decides to try some of the new products, but Bill decides not to buy any. Because Bob and Bill are so similar, any difference in their spending behaviour after the promotion is precisely the effect of the promotion.

This is how matching methods try to measure the causal effect:

1. For every participating customer, we find a matching twin customer.
2. The promotion effect is then the average difference between each pair of customers.

### Propensity score matching

Usually we find a matching observation by looking at a set of covariates $$X$$. In general, it is very hard to find an *exact* match for every customer, especially if $$X$$ is of high dimensionality.

*Propensity score matching* is a simplification of the matching procedure. Instead of matching directly on the covariates $$X$$, we match on the individual propensity scores $$p$$. The propensity score is the likelihood (or probability) that an observation receives treatment (a customer participates in the promotion).

Propensity score matching works as follows:

1. Estimate a probabilistic model that predicts the likelihood of receiving treatment based on some set of pre-treatment covariates $$X$$. We can use any probabilistic model for this, but most commonly generalized linear models (GLMs) or generalized additive models (GAMs) are used.
2. For every observation in the treatment group, find a similar observation in the control group based on their propensity score $$p$$.
3. Estimate the treatment effect by comparing the treatment group with the group of matched control observations.

### Turning theory into practice

Now we have processed that the theory, it is finally time to get our hands dirty and see how these methods work in practice! We'll work with the LaLonde[^1] data set.

### References

[^1]: Dehejia, Rajeev and Sadek Wahba. 1999. *Causal Effects in Non-Experimental Studies: Re-Evaluating the Evaluation of Training Programs*, Journal of the American Statistical Association 94 (448): 1053-1062.
