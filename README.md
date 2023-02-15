# fairness-optimization: Recommender Systems and Fairness Metrics

This repository contains code for implementing the methods outlined in the research article, "Fairness in
Collaborative-Filtering Recommender Systems". The article highlights the issue of fairness in collaborative-filtering
recommender systems, which can be affected by historical data bias. Biased data can lead to unfair recommendations for
users from minority groups.

In this work, we propose four new fairness metrics that address various types of unfairness, as existing fairness
metrics have proven to be insufficient. These new metrics can be optimized by adding fairness terms to the learning
objective. We have conducted experiments on synthetic and real data, which show that our proposed metrics outperform the
baseline, and the fairness objectives effectively help minimize unfairness.

The repository contains the following files:

fairness_metrics.py: This file contains the implementation of the four new fairness metrics proposed in the article.

[models](models): This file contains the implementation of the new recommender system models that use
fairness-optimization.

[methods.py](fairness_methods%2Fmethods.py).py: This file contains the code for each fairness metrics from the article.

[article_recovery.ipynb](article_recovery.ipynb): This jupyter notebook recovery the article results on movieLen data &
synthetic data.

To use this repository, you can download or clone it to your local machine.
The recommended way to run the code is to create a virtual environment using Python 3.9 or higher.
Then, install the required dependencies by running pip install -r requirements.txt.

After installing the dependencies, you can run the [article_recovery.ipynb](article_recovery.ipynb).
This script will train the basic model, optimize the fairness metrics, and evaluate the model on the test data.

We hope that this repository will be useful for researchers and practitioners who are interested in developing more fair
collaborative-filtering recommender systems. Please feel free to reach out to us with any questions or feedback.

## resources

Article: ["Beyond Parity:Fairness Objectives for Collaborative Filtering"](https://arxiv.org/pdf/1705.08804.pdf)

Explanation:

1. [Youtube link](https://www.google.com/search?q=code+for+Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering&rlz=1C5GCEM_enIL1032IL1032&sxsrf=ALiCzsZACsJifJOqejrgmEyimTqkxszmNw%3A1671954851128&ei=owGoY9G7B8GW9u8PzPWjyAo&ved=0ahUKEwiRme2XpZT8AhVBi_0HHcz6CKkQ4dUDCA8&uact=5&oq=code+for+Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQA0oECEEYAEoECEYYAFAAWABg_AFoAHABeACAAX-IAX-SAQMwLjGYAQCgAQHAAQE&sclient=gws-wiz-serp#fpstate=ive&vld=cid:f93e384b,vid:uMApSkGGQKs)
2. [Lecture](https://www.google.com/search?q=Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering%3A&oq=Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering%3A&aqs=chrome..69i57j35i39l2j0i22i30.871j0j7&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:9fe4e62d,vid:9EWSuoNqBQo)

## metrics

1. val_score:
   "measures inconsistency in signed estimation error across the user types"-
   The concept of value unfairness refers to situations where one group of users consistently receives higher or lower
   recommendations than their actual preferences. If there is an equal balance of overestimation and underestimation,
   or if both groups of users experience errors in the same direction and magnitude, then the value unfairness is
   minimized. However, when one group is consistently overestimated and the other group is consistently underestimated,
   then the value unfairness becomes more significant. A practical example of value unfairness in a course
   recommendation system could be male students being recommended STEM courses even if they are not interested in such
   topics, while female students are not recommended STEM courses even if they are interested in them.

$$
U_{val} = 1/n \sum_{j=1}^{n} \left |(E_{g}   [y \right ]_{j} - E_{g} \left [r \right ]_{j}) - (E_{\neg g}
\left [y \right ]_{j} - E_{\neg g} \left [r \right ]_{j})|
$$

2. abs_score:
   "measures inconsistency in absolute estimation error across user types"

$$U_{abs} = \frac{1}{n} \sum_{j=1}^{n} \left| \left|E_{g}[y]{j} - E{g}[r]{j}\right| - \left|E{\neg g}[y]{j} - E{\neg
g}[r]_{j}\right| \right|$$

3. under_score:
   "measures inconsistency in how much the predictions underestimate the true ratings"

$$U_{under} = \frac{1}{n} \sum_{j=1}^{n} \left| \max \left(0, E_{g}[r]{j} - E{g}[y]{j}\right) - \max \left(0, E{\neg
g}[r]{j} - E{\neg g}[y]_{j}\right) \right|$$

4. over_score:
   "measures inconsistency in how much the predictions overestimate the true ratings"

$$U_{over} = \frac{1}{n} \sum_{j=1}^{n} \left| \max \left(0, E_{g}[y]{j} - E{g}[r]{j}\right) - \max \left(0, E{\neg
g}[y]{j} - E{\neg g}[r]_{j}\right) \right|$$

## models
