
import string


# TODO: add some more metrics here for the new tasks.

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains'}
default_metric = 'em'


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction




