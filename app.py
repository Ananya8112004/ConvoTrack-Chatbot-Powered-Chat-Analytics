import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from chatbot import chatbot_ui
import helperDemo  # your chatbot logic
import preprocessorDemo  # for chat preprocessing
import pandas as pd
import streamlit.components.v1 as components
from preprocessor import preprocess_chat
from helper import create_index, search_messages


def add_bg_with_sidebar_color():
    st.markdown(
         f"""
         <style>
         /* Sidebar background color */
         section[data-testid="stSidebar"] {{
             background-color: #99ff99; /* Light Green */
         }}

         /* App background image (optional, can remove if you don't want) */
         .stApp {{
             background-image: url("https://wallpapers.com/images/featured/light-green-a9i3jcgdgez0iyyd.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}

         /* Sidebar text color */
         .css-10trblm, .css-1cpxqw2 {{
             color: black;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_with_sidebar_color()


st.sidebar.title("ConvoTrack")

# Upload chat file
uploaded_file = st.sidebar.file_uploader("Choose a file")

with st.sidebar:
    if uploaded_file is not None:
        try:
            # Decode uploaded text
            bytes_data = uploaded_file.getvalue()
            data = bytes_data.decode("utf-8")

            # Preprocess chat data
            df = preprocessorDemo.preprocess(data)
            st.success("✅ Chat uploaded.")

            # Chatbot interface
            st.subheader("💬 Ask Cohere")
            prompt = st.text_input("Your question", placeholder="e.g., What day was the most active?")

            if prompt:
                with st.spinner("Thinking..."):
                    try:
                        # Prepare context
                        context = "\n".join((df['Sender'] + ": " + df['Message']).tolist())
                        context = context[:10000]

                        # Call Cohere
                        model = helperDemo.init_cohere()
                        response = helperDemo.ask_cohere(model, prompt, context)

                        st.markdown("**🧠 Cohere's Answer:**")
                        st.write(response)

                    except Exception as e:
                        st.error(f"❌ Error from Cohere: {e}")
        except Exception as e:
            st.error(f"❌ File processing failed: {e}")




# Main Analysis Screen
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)


    if st.sidebar.button("Show Analysis"):


        # stats area
        num_messages, words, num_media_messages, links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(len(words))
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Total Links")
            st.title(links)

        # app.py

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig = helper.plot_monthly_timeline(timeline)
        st.pyplot(fig)

        #daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='violet')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title("Activity Map")
        col1,col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='magenta')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig,ax = plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='pink')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        import plotly.express as px

        if selected_user == 'Overall':
            st.title('📊 Most Busy Users')
            x, new_df = helper.most_busy_users(df)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    x=x.index,
                    y=x.values,
                    labels={'x': 'Users', 'y': 'Message Count'},
                    title='Top Active Users',
                    text=x.values,
                    color=x.values,
                    color_continuous_scale='Reds'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(new_df.style.highlight_max(axis=0, color='salmon'))




        # Forming word cloud
        st.title("Word Cloud")
        df_wc = helper.create_word_cloud(selected_user, df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)





        # most common words
        import plotly.express as px

        st.title("🗣️ Most Common Words")

        most_common_df = helper.most_common_words(selected_user, df)

        # Rename for clarity if needed
        common_words_df = most_common_df.rename(columns={0: 'Word', 1: 'Count'})

        # Create Plotly bar chart
        fig = px.bar(
            common_words_df.sort_values(by='Count', ascending=True),  # smallest to largest for horizontal layout
            x='Count',
            y='Word',
            orientation='h',
            text='Count',
            color='Count',
            color_continuous_scale='tealrose',
        )

        # Update layout for cleaner visuals
        fig.update_layout(
            title=dict(text='Top Words Used in Chat 💬', x=0.5, font=dict(size=20, color='black')),
            xaxis_title='Frequency',
            yaxis_title='Words',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            margin=dict(l=50, r=30, t=60, b=40),
            height=500,
        )

        # Value labels position
        fig.update_traces(textposition='auto', marker_line_width=0.5, marker_line_color='darkgray')

        # Show chart
        st.plotly_chart(fig, use_container_width=True)





        # emojis analysis
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.title("😊 Most Common Emojis")

        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(
                emoji_df.style.set_properties(**{'text-align': 'center'}).highlight_max(axis=0, color='lightyellow'))

        with col2:
            # Limit to top 10 emojis for clarity
            top_emojis = emoji_df.head(10)

            fig, ax = plt.subplots(figsize=(6, 6))

            # Colors from Seaborn palette
            colors = sns.color_palette('pastel')[0:len(top_emojis)]

            # Create donut chart
            wedges, texts, autotexts = ax.pie(
                top_emojis[1],
                labels=top_emojis[0],
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.4)  # Donut shape
            )

            # Style texts
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_color('black')
                autotext.set_weight('bold')

            ax.set_title("Top 10 Emojis Used", fontsize=16, fontweight='bold')
            st.pyplot(fig)





        # daily sentiment area
        import plotly.graph_objects as go
        import pandas as pd

        st.title("📅 Daily Sentiment and Mood Tracker")

        # Fetch sentiment DataFrame
        daily_df = helper.daily_sentiment(df, selected_user)

        # Display styled table
        st.dataframe(daily_df.style.set_properties(**{
            'background-color': '#fdf6e3',
            'color': '#000',
            'border-color': 'gray'
        }).highlight_max(axis=0, color='lightgreen'))

        # Plotly Area Chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_df['only_date'],
            y=daily_df['sentiment'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='royalblue', width=3),
            hoverinfo='x+y',
            name='Sentiment Score'
        ))

        fig.update_layout(
            title=dict(text='📈 Sentiment Over Time', x=0.5, font=dict(size=24, color='darkblue')),
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            template='plotly_white',
            hovermode='x unified',
            height=500,
            margin=dict(t=60, l=40, r=40, b=40),
            font=dict(family='Arial', size=14)
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)


        if uploaded_file is not None:
            # 📈 Mood swings chart
            st.title("📈 Mood Swings Over Time")
            helper.plot_mood_swings(daily_df)

            # ☁️ WordCloud for Happy Days
            st.title("☁️ WordCloud for Happy Days")
            helper.generate_wordcloud(daily_df, "Happy")

            # ☁️ WordCloud for Sad Days
            st.title("☁️ WordCloud for Sad Days")
            helper.generate_wordcloud(daily_df, "Sad")

            # 🥰 Top Happiest and Saddest Days
            st.title("🥰 Top Happiest and Saddest Days 😢")
            helper.show_top_moods(daily_df)

            st.write(daily_df.columns)  # Debugging line to check the columns

