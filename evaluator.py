from metrics import *

def evaluate_output(
    prediction,
    reference
):

    results = {
        "bertscore": bertscore_metric(
            prediction,
            reference
        ),

        "bleu": bleu_metric(
            prediction,
            reference
        ),

        "chrf": chrf_metric_score(
            prediction,
            reference
        ),

        "language_fidelity": language_fidelity(
            prediction
        ),

        "code_switch_rate": code_switch_rate(
            prediction
        )
    }

    return results