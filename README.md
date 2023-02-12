# fairness-optimization

## resources

Article: ["Beyond Parity:Fairness Objectives for Collaborative Filtering"](https://arxiv.org/pdf/1705.08804.pdf)

Explanation:

1. [Youtube link](https://www.google.com/search?q=code+for+Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering&rlz=1C5GCEM_enIL1032IL1032&sxsrf=ALiCzsZACsJifJOqejrgmEyimTqkxszmNw%3A1671954851128&ei=owGoY9G7B8GW9u8PzPWjyAo&ved=0ahUKEwiRme2XpZT8AhVBi_0HHcz6CKkQ4dUDCA8&uact=5&oq=code+for+Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQA0oECEEYAEoECEYYAFAAWABg_AFoAHABeACAAX-IAX-SAQMwLjGYAQCgAQHAAQE&sclient=gws-wiz-serp#fpstate=ive&vld=cid:f93e384b,vid:uMApSkGGQKs)
2. [Lecture](https://www.google.com/search?q=Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering%3A&oq=Beyond+Parity%3A+Fairness+Objectives+for+Collaborative+Filtering%3A&aqs=chrome..69i57j35i39l2j0i22i30.871j0j7&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:9fe4e62d,vid:9EWSuoNqBQo)

## metrics

1. val_score:
   "measures inconsistency in signed estimation error across the user types"

   $$
   U_{val} = 1/n \sum_{j=1}^{n} \left |(E_{g}   [y \right ]_{j} - E_{g}  \left [r  \right ]_{j}) - (E_{\neg g}  \left [y \right ]_{j} - E_{\neg g}  \left [r  \right ]_{j})|
   $$
2. abs_score:
   "measures inconsistency in absolute estimation error across user types"

$$U_{abs} = \frac{1}{n} \sum_{j=1}^{n} \left| \left|E_{g}[y]{j} - E{g}[r]{j}\right| - \left|E{\neg g}[y]{j} - E{\neg g}[r]_{j}\right| \right|$$
3. under_score:
   "measures inconsistency in how much the predictions underestimate the true ratings"

$$U_{under} = \frac{1}{n} \sum_{j=1}^{n} \left| \max \left{0, E_{g}[r]{j} - E{g}[y]{j} \right} - \max \left{0, E{\neg g}[r]{j} - E{\neg g}[y]_{j} \right} \right|$$
4. over_score:
   "measures inconsistency in how much the predictions overestimate the true ratings"

$$U_{over} = \frac{1}{n} \sum_{j=1}^{n} \left| \max \left{0, E_{g}[y]{j} - E{g}[r]{j} \right} - \max \left{0, E{\neg g}[y]{j} - E{\neg g}[r]_{j} \right} \right|$$


## models
