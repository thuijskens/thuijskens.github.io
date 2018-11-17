---
layout: post
title: Data scientists, the only useful code is production code
---

The responsibilities of a data scientist can be very diverse, and people have written in the past about the different types of data scientists that exist in the industry. The types of data scientists range from a more analyst-like role, to more software engineering-focused roles. It is partly due to the different responsibilities those jobs require, and the diverse backgrounds data scientists come from, that they sometimes have a bad reputation amongst peers when it comes to writing good quality code. Not everybody comes to data science with a software engineering background.

Regardless of what the responsibilities of a data scientist are, code is a main (by)product of his or her work. Whether the scientist is producing ad-hoc analyses for a business stakeholder, or building a machine learning model sitting behind a [RESTful API](https://en.wikipedia.org/wiki/Representational_state_transfer), the main output is always code. Since most data scientists don't come from a software engineering background, the quality of that code can vary a lot, causing issues with reproducibility and maintainability later down the line.

<!--excerpt-->

When it comes to poor coding quality, some data scientists will say that their work does not touch a production system, and that their code therefore does not need to be of a high standard. However, I would argue that common outputs of a data scientist's work can actually be considered production:

- The ad-hoc analysis that discusses a useful insight that was shown to a senior executive may be used for a key financial decision. You may want to re-run that analysis in the future, and you can't tell him or her a month later that you can't reproduce the analysis because your codebase is incomprehensible.
- The report that gets sent out every week to a whole business unit. Multiple teams will use that to base decisions on, so you would want the code that generates it to be well-tested.
- The modelling pipeline you wrote that dumps scores daily into a CRM database. If your model gets enough traction, the business will want to roll it out to other teams. Other people now suddenly need to be able to read, extend and execute your codebase.

Production code is any code that feeds some business (decision) process. Since data science by design is meant to affect business processes, most data scientists are in fact writing code that can be considered production. Data scientists should therefore always strive to write good quality code, regardless of the type of output they create. Whatever type of data scientist you are, the code you write is only useful if it is **production code**.

## Production code

It is hard to give a general definition of what production code is, but a key difference with non-production code, is that production code gets read and executed by many other people, instead of just the person that wrote it. We should therefore aim for our code to be

  - **Reproducible**, because many people are going to run it.
  - **Modular** and **well-documented**, because many people are going to read it.

These are challenges the software engineering world has already encountered, and it helps to look at how this field tackles them. I'll discuss some tools that can give you an immediate positive impact on the quality of your work (if you are data scientist) or the quality of your team (if you are a data science manager).

Some of these tools may seem daunting to learn initially, but for a lot of these you can copy templates that you create for your first project, to your other projects. All it takes therefore is a one-time investment to learn some useful tools and paradigms, that will pay dividends throughout your career as a data scientist.

To help you get started with these tools, I have set up a [bare-bones repository](https://github.com/thuijskens/production-tools) that contains basic template files for some of the tools that I will discuss.

## Reproducible code

When you setup the codebase for your shiny new data science project, you should immediately set up the following tools:

- **Version control** your codebase using `git` or a similar tool.
  - The first thing you should do is to set up a version controlled repository on a remote server, so that each team member can pull an up-to-date version of the code. A great, 5 minute introduction to `git` can be found [here](http://rogerdudler.github.io/git-guide/).
  - Try to push code changes to the remote at a regular frequency (I would recommend daily, if possible).
  - **Do not work on a single branch**, whether you work alone or in a team. Choose a [git branching workflow](https://www.atlassian.com/git/tutorials/comparing-workflows) you like (it doesn't really matter which one, just use one!) and stick with it.
- Create a **reproducible python environment** with `virtualenv` or `conda`.
  - These tools take a configuration file (a `requirements.txt` in case of `virtualenv`, or a `environment.yml` in case of `conda`) that contains a list of the packages (with version numbers!).
  - Put this file in version control and distribute it across your team to ensure everybody is working in the same environment.
  - Consider coming up with a standard base environment so that you can reuse that whenever you or a team member start a new project.
  - **Example**: See the git repo [here](https://github.com/thuijskens/production-tools/blob/master/requirements.txt).
- **Drop Jupyter notebooks** as your **main** development tool.
  - Jupyter notebooks are great for quick exploration of the data you are working with, but do not use them as your main development tool.
  - Notebooks do not encourage a reproducible workflow, and you should see this [talk](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g362da58057_0_1) for a good overview of why they don't.
  - Use a proper IDE like PyCharm or VS code (or vim if you're into that) when developing code. Convince your employer to buy you professional editions of this software (this is usually peanuts for the company, and can be a massive productivity boost). I develop most of my code locally, but use PyCharm's remote execution to execute any code on the cloud or an internal VM.

## Well-documented code

After you have set up your project in a way that will support reproducibility, take the following steps to ensure that it is possible for other people to read and understand it.

- Adopt a **common project structure**.
  - A common structure will make it easy for both members of your team, as well as other colleagues, to understand your codebase.
  - The specifics of the project structure again don't matter much, just choose one and stick with it. The below are great starting points. The templates from [Cookiecutter](http://drivendata.github.io/cookiecutter-data-science/) and [Satalia](https://github.com/Satalia/production-data-science) are great starting points.
- Choose a **coding style convention**, and configure a linter to enforce it (potentially pre-commit).
  - Enforcing code conventions will make it easier for other people to read your codebase. I would recommend using something like [PEP8](https://www.python.org/dev/peps/pep-0008/), as many people in industry will already be familiar with it.
  - Enforcing coding conventions using a pre-commit linter can be good, as the programmer will not have to worry too much about the conventions during programming, because the linter will pick it up.
  - Using a linter will avoid pull requests (PRs) that are littered with coding style comments. These PRs are the worst to both review and receive a review for.
  - *Example*: [black pre-commit plugin](https://github.com/ambv/black) or [yapf](https://github.com/google/yapf).
- **Use Sphinx** to automatically create the documentation of your codebase.
  - Pick a docstring format. I personally prefer [NumPyDoc](https://github.com/numpy/numpydoc), but there are others. Again it does not matter which format you choose, just choose one and stick with it. Configure your IDE to use that docstring format, so that it will automatically create a template when you write a new function or class.
  - Use `sphinx-quickstart` to get a set of out-of-the-box configuration files, or copy the ones from [my repository](https://github.com/thuijskens/production-tools/tree/master/docs).
  - Using Sphinx can seem daunting at first, but it is one of those things that you set up once and then copy the default configuration files around for from project to project.

## Modular code

Finally, follow the below steps to ensure your codebase can be executed easily and robustly:

- Use a **pipeline framework** for your engineering and modelling workflows.
  - Frameworks like [Apache Airflow](https://airflow.apache.org/) and [Luigi](https://luigi.readthedocs.io/en/stable/) are a great way to make your code inherently modular.
  - They allow you to build your workflow as a series of nodes in a graph, and usually gives you things like dependency management and workflow execution for free.
- Write **unit tests** for your codebase.
  - Pick a unit testing framework (like [nose](https://nose.readthedocs.io/en/latest/) or [pytest](https://docs.pytest.org/en/latest/)) and stick with it.
  - Writing unit tests can be cumbersome, but you want these tests in your codebase to ensure everything behaves as expected! This is especially important in data science, where we deal a lot with black-box algorithms.
- Consider adding **continuous integration** (CI) to your repository.
  - CI can be used to run your unit tests or pipeline after every commit or merge, making sure that no change to the codebase breaks it.
  - Many vendors offer integration with the code hosting platforms like GitHub or GitLab. All you need typically is a [configuration file](https://github.com/thuijskens/production-tools/blob/master/.circleci/config.yml) that is committed to your codebase, and you are ready to go!

Finally, ensure that the environment you develop your code in is reasonably similar to the production environment the code is going to run in. Especially in companies where development, staging and production environments for data science are not yet well-defined, I have seen teams developing code on architecture that is extremely different than the architecture the code actually has to run on in the end.

**Data scientists**, adopt these standards and see your employability increase, and complaints by your more software engineering-focused colleagues decrease. You'll spend less time worrying about reproducibility, and rewriting software so that it can make it to production. The time saved here can be used to focus more on the fun part of our job: building models.

**Data science managers**, consider giving your team members a couple of days to get up to speed with these tools, and you will see that your codebases become more stable. It will be easier to onboard new members to your team and you will spend less time translating initial insights to production pipelines. Having a common way of working will also allow your team to start building utilities that tap into these conventions, increasing the overall productivity of your team.
