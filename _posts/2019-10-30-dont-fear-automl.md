---
layout: post
title: Don't fear the rise of automated machine learning
---

There are a wealth of high-quality research tools available in the machine learning open source community. However, as an industry we still lack standardised tooling that helps us put models into production. A lot of the code we produce can be repetitive, and we are still lacking industry-wide standards for things like storing experiment results, building and versioning models, and tracking model performance over time in production.

In the last few years, there have been a number of start-ups and initiatives addressing these issues. In particular, there has been a noticeable rise in the number of companies (Google's AutoML, DataRobot, SparkBeyond, SigOpt) and open-source solutions ([AutoWEKA](https://www.cs.ubc.ca/labs/beta/Projects/autoweka/), [auto-sklearn](https://github.com/automl/auto-sklearn), [SMAC](https://github.com/automl/SMAC3)) that provide automatic machine learning solutions (in production).

In itself this is a great marker of the professionalisation of machine learning, and it helps democratise the benefits machine learning can bring to a business. As more standardised solutions become available on the market, the technical know-how that is needed to deploy machine learning algorithms becomes more accessible to our colleagues in engineering and colleagues in non-technical disciplines.

Sometimes, these products are however advertised as making the job of a data scientist completely redundant, and that all you need to apply machine learning to your business successfully is a person that knows how to call an API. This can be especially confusing to people with no industry experience. They often believe that data science only covers algorithmic modelling, and therefore can be fully automated. In reality, data scientists should not fear being made redundant anytime soon. Rather, they should welcome the advancements of automated machine learning methods, and the productivity boost they will bring to their workflow.

Automated machine learning methods will not replace specialised experts because the set of tasks they solve are only a relatively small part of a data scientist's overall workflow. A data scientist has a number of engineering-focussed tasks, such as collecting data, building data engineering pipelines, and designing algorithms, some of which can potentially be automated. However, as a data scientist you are also often a consultant to the wider business. With that comes a range of challenges that is less often discussed when we talk about the job of a data scientist. In most non-data-driven businesses (and therefore most businesses), a lot of time is spent on convincing the business of the merits of your model, setting up the right infrastructure for your software to run on, scoping of projects and holding constant communication updates with all the relevant stakeholders. None of the latter set of responsibilities can be automated by machine learning.

## Life in the machine

To understand this a bit better, we can look at an average project a data scientist may work on in his or her career. Imagine a data scientist, called Sarah, working for a company that provides car loans. Customers are charged an interest rate for the principal value on the loan, and this interest rate is set by the internal pricing team. Her task is to deploy a pricing algorithm, that recommends the optimal interest rate for a customer, based on the available characteristics for that customer. During the project, she'll have to iterate through the following steps:

- **Defining the problem**: Algorithmic pricing falls in the domain of causal analysis, since you are trying to estimate the causal effect of making a price change. The pricing and demand mechanism are often confounded, and Sarah will need to define the mathematical problem in such a way that she deals with this appropriately. Additionally, Sarah needs to understand if there is any heterogeneity in products the company offers, and what constraints her final algorithm should satisfy.
- **Collecting relevant data**: Sarah will have to work with the internal IT team, to obtain a dataset of all loan offers given out by the company that she can use to train her machine learning algorithm on.
- **Cleaning the data**: Once she has received that data, she'll need to build data engineering pipelines that clean up any faults in the data, and transform the data into a format suitable for her algorithm.
- **Selecting the objective function**: Business teams often try to optimise for a number of different metrics. In pricing optimisation, we have to combine these metrics into a single objective function that we want to optimise. Defining the single objective function with business teams, often requires frequent iteration.
- **Selecting a suitable algorithm**: As part of the pricing optimisation problem, Sarah will want to build a predictive algorithm that models the demand as a function of the price, as well as other explanatory variables.
- **Deploying the algorithm**: Once a model has been fitted, additional robust engineering pipelines will have to be built to munge the results of the algorithm into a format that can be consumed by the business, and to serve out the results.
- **Tracking the algorithm after deployment**: After deployment, continuous tracking of the performance of the algorithm needs to be set up to avoid the performance of the model degrading over time.

This process is not linear, but iterative, and requires Sarah to have constant touch points with the different business teams throughout the process.
Although automated machine learning methods promise to solve certain parts of this process, but clearly not all of the above tasks can be automated away by machine learning. Data scientists should therefore not fear to be made redundant any time soon.

## Rise of the machines

Instead, data scientists should welcome automated machine learning methods. Automated machine learning methods are part of the standardised tooling that our industry needs. These methods can help realise productivity gains in those areas of our workflow where they are applicable. As these methods become better over time and realise such productivity gains, the work of a data scientist will shift into a different direction.

Because model estimation and inference becomes cheaper, more time will be spent on translating the business problem correctly into a mathematical problem. Specialised loss function design will become more common. The pricing use-case discussed above is a great example where proper loss function design will make or break the successful deployment of the algorithm in the business. Another example is the WTE-RNN algorithm discussed [here](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/), where a simple change in loss function can be strikingly effective (allowing an RNN to learn from censored observations, inspired by survival analysis methods).

Additionally, I've found that a lot of the machine learning code a data scientist writes can be quite repetitive. In many projects we reuse the same kind of code to create cross-validation, grid search, and batch update loops. By handing over the repetitiveness of writing model estimation code to automation, we can spend more time on curating the data sets that we use for our models.

Automated machine learning methods will also help increase the trust in results obtained by machine learning methods. As argued by Sculley (2018)[^1], modern machine learning, has come to emphasise methods that yield impressive empirical results, but are hard to analyse theoretically. Standardising the way empirical evaluation is done, will help increase the quality of the work and research that is done in the machine learning field.

Finally, automated machine learning methods will enable engineers and analysts, with less technical knowledge than specialised data scientists, to apply machine learning to the problems in the business that represent low-hanging fruit. This has two benefits for data scientists. Firstly, it will give them time to focus on those use-cases that require deeper expertise. This represents both a gain for the business as a whole, as well as for a data scientist. For the business, this results in a better allocation of talent to problems, and it will allow data scientists to work on the more interesting and challenging problems (which is usually considered desirable by data scientists). Secondly, the number of successfully deployed machine learning use-cases in the business will increase, due to the higher number of people being able to work on such use-cases.

The latter benefit has an interesting side-effect. The more machine learning is successfully applied to the business, the more other departments will want a piece of the action. The democratisation of machine learning that these methods achieve, will actually increase the demand for data scientists even more. All the more reason for data scientists to fully embrace the rise of automated machine learning.

## References

[^1]: https://openreview.net/pdf?id=rJWF0Fywf

[^2]: https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf
