from datasets import load_dataset


def load(cache_dir):

    dataset = load_dataset(
        "HuggingFaceM4/ChartQA",
        cache_dir=cache_dir
    )

    train_dataset = [format_data(sample) for sample in dataset["train"]]
    eval_dataset = [format_data(sample) for sample in dataset["val"]]
    test_dataset = [format_data(sample) for sample in dataset["test"]]

    return train_dataset, eval_dataset, test_dataset



def format_data(sample):

    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    return { "message":
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ]
    }