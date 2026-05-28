from datasets import load_dataset
from config import NUM_SAMPLES

def build_xquad_dataset():

    en_data = load_dataset("xquad", "xquad.en")["validation"]
    zh_data = load_dataset("xquad", "xquad.zh")["validation"]
    vi_data = load_dataset("xquad", "xquad.vi")["validation"]

    samples = []

    for i in range(NUM_SAMPLES):

        sample = {
            "question": vi_data[i]["question"],
            "reference_answer": vi_data[i]["answers"]["text"][0],
            "en_context": en_data[i]["context"],
            "zh_context": zh_data[i]["context"]
        }

        samples.append(sample)

    return samples