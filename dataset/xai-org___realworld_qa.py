from datasets import load_dataset


def load(cache_dir):

    dataset = load_dataset(
        "xai-org/RealworldQA",
        cache_dir=cache_dir
    )

    dataset = [format_data(sample) for sample in dataset["test"]]

    train_dataset = dataset[:612]
    eval_dataset = dataset[612:689]
    test_dataset = dataset[689:]

    return train_dataset, eval_dataset, test_dataset



def format_data(sample):

    system_message = """You are a Vision Language Model specialized in interpreting visual data from autonomous driving scenarios, using only camera images.
    Your task is to analyze the provided camera image and respond to queries with concise answers, usually a single word, number, or short phrase.
    The image contains information about the surrounding environment, including roads, vehicles, pedestrians, traffic signs, and other relevant objects.
    Focus on delivering accurate, succinct answers based on the visual information from the camera image and your understanding of autonomous driving scenarios. Avoid additional explanation unless absolutely necessary."""

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
                        # "max_pixels": 1280 * 28,
                    },
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
    }