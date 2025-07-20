# from app import num_messages
from turtle import st
import seaborn as sns
from matplotlib import pyplot as plt
from urlextract import URLExtract
from wordcloud import wordcloud, WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

def fetch_stats(selected_user,df):
    if selected_user!='Overall':
        df = df[df['user'] == selected_user]

    # fetch no. of messages
    num_messages = df.shape[0]

    # fetch total no. of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    #fetch no. of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch no. of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, words, num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'user':'name','count':'percent'})
    return x,df

def create_word_cloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(10))
    return emoji_df

# In helper.py

import matplotlib.pyplot as plt
import seaborn as sns

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

def plot_monthly_timeline(timeline_df):
    sns.set_style("whitegrid")
    sns.set_palette("coolwarm")  # Color style

    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=timeline_df, x='time', y='message', marker='o', linewidth=2.5)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.title('Monthly Messages Timeline 📅', fontsize=20, color='darkblue', pad=20)
    plt.xlabel('Time (Month-Year)', fontsize=14, color='purple')
    plt.ylabel('Number of Messages', fontsize=14, color='purple')

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    return fig


def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)
    return user_heatmap


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def daily_sentiment(df, selected_user):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()

    for message in df['message']:
        if len(message.split()) > 2:
            score = analyzer.polarity_scores(message)['compound']
            sentiments.append(score)
        else:
            sentiments.append(0.0)

    df['sentiment'] = sentiments

    daily_sentiment = df.groupby('only_date')['sentiment'].mean().reset_index()

    def get_mood(score):
        if score >= 0.75:
            return 'Ecstatic'
        elif score >= 0.5:
            return 'Happy'
        elif score >= 0.1:
            return 'Content'
        elif score > -0.1:
            return 'Neutral'
        elif score > -0.5:
            return 'Sad'
        elif score > -0.75:
            return 'Angry'
        else:
            return 'Depressed'

    daily_sentiment['mood'] = daily_sentiment['sentiment'].apply(get_mood)

    return daily_sentiment


import streamlit as st  # Import Streamlit in helper.py


def generate_wordcloud(df, mood_type):
    """Generate and display a wordcloud for a specific mood type."""
    mood_df = df[df['mood'] == mood_type]

    # Generate text using the 'mood' column instead of 'message'
    text = " ".join(mood_df['mood'].dropna())

    if text.strip() == "":
        st.warning(f"No messages found for mood: {mood_type}")
        return

    wc = WordCloud(width=500, height=300, min_font_size=10, background_color='white').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)


def plot_mood_swings(daily_df):
    import matplotlib.pyplot as plt
    import streamlit as st

    fig, ax = plt.subplots()
    ax.plot(daily_df['only_date'], daily_df['sentiment'], color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Mood Swings Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def show_top_moods(daily_df):
    """Show happiest and saddest days."""
    happiest = daily_df.loc[daily_df['sentiment'].idxmax()]
    saddest = daily_df.loc[daily_df['sentiment'].idxmin()]

    st.subheader("🌟 Happiest Day")
    st.write(f"Date: {happiest['only_date']} - Mood: {happiest['mood']} - Score: {happiest['sentiment']:.2f}")

    st.subheader("😢 Saddest Day")
    st.write(f"Date: {saddest['only_date']} - Mood: {saddest['mood']} - Score: {saddest['sentiment']:.2f}")




from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model and create FAISS index from chat messages
def create_index(messages):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(messages)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, model

# Search for semantically similar messages

def search_messages(query, model, index, df, top_k=5):
    query_embedding = model.encode([query])
    _, result_indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[result_indices[0]]




import cohere
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

def init_cohere():
    return cohere.Client(api_key)

def ask_cohere(model, prompt, context):
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    try:
        response = model.generate(
            model='command-r',
            prompt=full_prompt,
            max_tokens=300
        )
        return response.generations[0].text
    except Exception as e:
        return f"Cohere API error: {e}"
