import pandas as pd
import matplotlib.pyplot as plt
import cv2

def plot_prompt_lenght(df):
    prompt_lengths = df["prompt"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6, 6))
    plt.hist(prompt_lengths)
    plt.title("Distribution of Lengths")
    plt.xlabel("Prompt Length")
    plt.show()

def image_id2path(img_id, folder):
    return f"/content/{folder}/images/{img_id}.png"


def show_images_and_prompts(df, folder, n):
    for ind, row in df[:n].iterrows():
        img_id = row["imgId"]
        prompt = row["prompt"]
        path = image_id2path(img_id, folder)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if ind % 2 == 0:
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
        else:
            plt.subplot(1, 2, 2)

        plt.imshow(image)
        list_prompt_words = prompt.split()
        if len(prompt) > 100:
            _len = len(list_prompt_words)
            prompt = "{}\n{}\n{}".format(
                " ".join(list_prompt_words[:_len // 3]),
                " ".join(list_prompt_words[_len // 3: 2 * _len // 3]),
                " ".join(list_prompt_words[2 * _len // 3:]),
            )
        elif len(prompt) > 50:
            _len = len(list_prompt_words)
            prompt = "{}\n{}".format(
                " ".join(list_prompt_words[:_len // 2]),
                " ".join(list_prompt_words[_len // 2:])
            )
        plt.title(prompt, fontsize=14)
        plt.axis("off")