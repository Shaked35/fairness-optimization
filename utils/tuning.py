import random


def get_random_hp_choises():
    hp_options = {
        "embed_dim": [5, 8, 12],
        "lr": [0.008, 0.3],
        "early_stop": [1e-7, 1e-5],
        "unfairness_reg": [0.2, 0.7], 'model': ['gmf', 'nmf'],
        "optimizar": ['sgd', 'adam']
    }
    final_choise = {}
    for name, min_max in hp_options.items():
        if isinstance(min_max[0], float):
            final_choise[name] = random.uniform(min_max[0], min_max[1])
        else:
            final_choise[name] = random.choice(min_max)
    return final_choise
