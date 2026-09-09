from matplotlib.transforms import Bbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tag_to_word(sentence, predictions):
    """
    predictions: list of tags (0 = none, 1 = start of aspect term, 2 = inside/mark of aspect term)
    sentence: list of words
    """
    # FIX: the original loop checked predictions[i] (the start index, always 1)
    # instead of predictions[j] (the token actually being examined), so the
    # "extend the term" branch was unreachable and every aspect term collapsed
    # to a single word. The manual `i = j` reassignment also did nothing, since
    # `for i, word in enumerate(sentence)` overwrites i on every outer
    # iteration regardless. This version correctly walks forward from a
    # "start of aspect" token and appends any following "inside aspect" (2)
    # tokens to the same term.
    terms = []
    i = 0
    while i < len(sentence):
        if predictions[i] == 1:
            w = sentence[i]
            j = i + 1
            while j < len(sentence) and predictions[j] == 2:
                w += ' ' + sentence[j]
                j += 1
            terms.append(w)
            i = j
        else:
            i += 1

    return terms

def tag_to_word_df(df, column_name, tags):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms_list = []
    for i in range(len(df)):
        sentence = df.iloc[i]['Tokens']
        sentence = sentence.replace("'", "").strip("][").split(', ')
        terms = tag_to_word(sentence, tags[i])
        terms_list.append(terms)
    df[column_name] = terms_list
    return df


def word_cloud (data):
    """
    create word cloud from sentence
    """
    from wordcloud import WordCloud
    wordcloud = WordCloud(collocations=False, 
                          background_color="cornflowerblue",
                          colormap="tab10",
                          max_words=50).generate(data)

    return wordcloud

def target_predicted_wordcloud(targets, predicted, file_name):
    """
    create word cloud from sentence with target and predicted
    """
    
    sns.set_theme(style='white', font_scale=1.5)
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    ax[0].imshow(word_cloud(targets))
    ax[0].axis("off")
    ax[0].set_title("Target")
    ax[1].imshow(word_cloud(predicted))
    ax[1].axis("off")
    ax[1].set_title("Predicted")
    fig.savefig(file_name, dpi=300, bbox_inches='tight')

def classification_report_read(report_path):
    """
    Read classification report from file
    """

    with open(report_path, 'r') as f:
        report = f.read()
    return report
    
def print_aligned(report1, report2, title1, title2):
    """
    print two classification report aligned to the columns
    """
    print (1*'\t', title1, 6*'\t', title2)
    report1 = report1.split('\n')
    report2 = report2.split('\n')
    for r1, r2 in zip(report1, report2):
        print(r1, '\t\t', r2)
