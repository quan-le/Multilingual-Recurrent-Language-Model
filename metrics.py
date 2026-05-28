from bert_score import score
from sacrebleu import sentence_bleu
from sacrebleu.metrics import CHRF
from langdetect import detect
import numpy as np

chrf_metric = CHRF()

def bertscore_metric(prediction, reference):

    P, R, F1 = score(
        [prediction],
        [reference],
        lang="vi"
    )

    return float(F1.mean())


def bleu_metric(prediction, reference):

    return sentence_bleu(
        prediction,
        [reference]
    ).score


def chrf_metric_score(prediction, reference):

    return chrf_metric.sentence_score(
        prediction,
        [reference]
    ).score


def language_fidelity(prediction, target_lang="vi"):

    try:

        detected = detect(prediction)

        return 1.0 if detected == target_lang else 0.0

    except:
        return 0.0


def code_switch_rate(prediction, target_lang="vi"):

    words = prediction.split()

    bad = 0

    for w in words:

        try:
            if detect(w) != target_lang:
                bad += 1
        except:
            bad += 1

    return bad / max(len(words), 1)