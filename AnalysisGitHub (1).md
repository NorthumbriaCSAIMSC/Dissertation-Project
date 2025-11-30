```python
import pandas as pd
import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
import empath
import warnings
warnings.filterwarnings("ignore")


nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
lexicon = empath.Empath()


df = pd.read_csv("ResponsesALL.csv", encoding="utf-8-sig")


df.columns = df.columns.str.replace('\ufeff', '')
print("Columns loaded:", df.columns)


def clean_text(text):
    text = str(text)
    text = text.replace("\x97", "-")  
    text = text.replace("\x92", "'")  
    text = text.replace("\x93", '"').replace("\x94", '"')  
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

df['response'] = df['response'].apply(clean_text)


def extract_features(text):
    doc = nlp(text)

    
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    n_sents = len(list(doc.sents))
    avg_sentence_length = n_tokens / (n_sents + 1)

    
    words = [t.lemma_.lower() for t in doc if t.is_alpha]
    type_token_ratio = len(set(words)) / (len(words) + 1)

    
    pos_counts = doc.count_by(spacy.attrs.POS)
    nouns = pos_counts.get(nlp.vocab.strings["NOUN"], 0)
    verbs = pos_counts.get(nlp.vocab.strings["VERB"], 0)
    adjectives = pos_counts.get(nlp.vocab.strings["ADJ"], 0)
    adverbs = pos_counts.get(nlp.vocab.strings["ADV"], 0)

    
    pronouns = [t.text.lower() for t in doc if t.pos_ == "PRON"]
    first_person = sum(p in ["i","me","my","we","our"] for p in pronouns)
    second_person = pronouns.count("you")

    
    n_questions = text.count("?")
    question_rate = n_questions / (n_sents + 1)

    
    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    
    flesch = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    
    empath_scores = lexicon.analyze(
        text,
        categories=["affection", "cognitive_mechanisms", "insight", "positive_emotion", "negative_emotion"],
        normalize=True
    )

    return {
        "n_tokens": n_tokens,
        "n_sents": n_sents,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "first_person_pronouns": first_person,
        "second_person_pronouns": second_person,
        "n_questions": n_questions,
        "question_rate": question_rate,
        "vader_pos": vader_scores["pos"],
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],
        "vader_compound": vader_scores["compound"],
        "tb_polarity": tb_polarity,
        "tb_subjectivity": tb_subjectivity,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": grade,
        "empath_affection": empath_scores["affection"],
        "empath_cognition": empath_scores["cognitive_mechanisms"],
        "empath_insight": empath_scores["insight"],
        "empath_pos_emotion": empath_scores["positive_emotion"],
        "empath_neg_emotion": empath_scores["negative_emotion"]
    }


feature_rows = []

for i, row in df.iterrows():
    text = row["response"]
    try:
        feats = extract_features(text)
        feats["vignette_id"] = row["vignette_id"]
        feats["model"] = row["model"]
        feats["modality"] = row["modality"]
        feature_rows.append(feats)
    except Exception as e:
        print(f"Error on row {i}: {e}")

features_df = pd.DataFrame(feature_rows)
print("Columns in features_df:", features_df.columns)


features_df.to_csv("features_output.csv", index=False, encoding="utf-8-sig")
print("Features saved to features_output.csv")

print("\n\n====== AVERAGE SENTIMENT BY MODEL ======")
print(features_df.groupby("model")["vader_compound"].mean())

print("\n\n====== AVERAGE SENTIMENT BY MODALITY ======")
print(features_df.groupby("modality")["vader_compound"].mean())

print("\n\n====== AVERAGE READABILITY (GRADE LEVEL) BY MODEL ======")
print(features_df.groupby("model")["flesch_kincaid_grade"].mean())

print("\n\n====== LEXICAL COMPLEXITY (TTR) BY MODEL ======")
print(features_df.groupby("model")["type_token_ratio"].mean())

print("\n\n====== EMPATH COGNITIVE vs EMOTIONAL CONTENT BY MODALITY ======")
print(features_df.groupby("modality")[["empath_cognition","empath_pos_emotion","empath_neg_emotion"]].mean())

```


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


sentiment_modality = features_df.groupby("modality")["vader_compound"].mean().sort_values()
plt.figure(figsize=(7,4))
sns.barplot(x=sentiment_modality.index, y=sentiment_modality.values, palette="viridis")
plt.ylabel("Average VADER Compound Sentiment")
plt.title("Average Sentiment by Modality")
plt.xticks(rotation=30)
plt.show()


readability_model = features_df.groupby("model")["flesch_kincaid_grade"].mean()
plt.figure(figsize=(5,4))
sns.barplot(x=readability_model.index, y=readability_model.values, palette="magma")
plt.ylabel("Flesch-Kincaid Grade Level")
plt.title("Average Readability by Model")
plt.show()


ttr_model = features_df.groupby("model")["type_token_ratio"].mean()
plt.figure(figsize=(5,4))
sns.barplot(x=ttr_model.index, y=ttr_model.values, palette="cool")
plt.ylabel("Type-Token Ratio")
plt.title("Lexical Diversity by Model")
plt.show()


empath_cols = ["empath_pos_emotion", "empath_neg_emotion"]
empath_modality = features_df.groupby("modality")[empath_cols].mean()

empath_modality.plot(kind="bar", figsize=(7,4), color=["#2ca02c","#d62728"])
plt.ylabel("Average Empath Score")
plt.title("Positive vs Negative Emotion by Modality")
plt.xticks(rotation=30)
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(data=features_df, x="flesch_kincaid_grade", y="vader_compound", hue="modality", palette="Set2", s=80)
plt.xlabel("Flesch-Kincaid Grade Level")
plt.ylabel("VADER Compound Sentiment")
plt.title("Sentiment vs Readability by Modality")
plt.legend(title="Modality")
plt.show()

```
