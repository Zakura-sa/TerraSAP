"""Cumulative-session accuracy metrics, independent of the training backend."""


def accuracy(y_pred, y_true, nb_old, increment=10):
    predictions, targets = list(y_pred), list(y_true)
    if len(predictions) != len(targets) or not targets:
        raise ValueError('Predictions and targets must have the same nonzero length')
    if increment <= 0:
        raise ValueError('Class increment must be positive')
    correct = [prediction == target for prediction, target in zip(predictions, targets)]

    def percentage(indices):
        if not indices:
            return 0.0
        hits = sum(1 for index in indices if correct[index])
        return round(100.0 * hits / len(indices), 2)

    result = {'total': percentage(list(range(len(targets))))}
    for start in range(0, max(targets) + 1, increment):
        indices = [i for i, label in enumerate(targets) if start <= label < start + increment]
        result[f'{start:02d}-{start + increment - 1:02d}'] = percentage(indices)
    result['old'] = percentage([i for i, label in enumerate(targets) if label < nb_old])
    result['new'] = percentage([i for i, label in enumerate(targets) if label >= nb_old])
    return result


def harmonic_accuracy(grouped, has_old=True):
    old, new = grouped['old'], grouped['new']
    if not has_old:
        return None, old, new
    score = 2.0 * old * new / (old + new) if old + new else 0.0
    return score, old, new
