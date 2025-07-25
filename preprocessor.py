import re
import pandas as pd
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]?[m]?\s-\s'

    messages = re.split(pattern, data)[1:]

    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # converting message data type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # to seperate user and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'(^[\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date

    df['year'] = df['date'].dt.year

    df['month_num'] = df['date'].dt.month

    df['month'] = df['date'].dt.month_name()

    df['day'] = df['date'].dt.day

    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour

    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    return df




import pandas as pd
import re

def preprocess_chat(chat_text):
    # Regex to match WhatsApp messages
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}[^ ]*) - (.*)$'
    messages = []
    current_message = ""
    current_date = ""
    current_time = ""

    for line in chat_text.split("\n"):
        match = re.match(pattern, line)
        if match:
            if current_message:
                messages.append([current_date, current_time, sender, current_message.strip()])
            current_date, current_time, message = match.groups()

            if ": " in message:
                sender, message = message.split(": ", 1)
            else:
                sender = "group_notification"
            current_message = message
        else:
            # Continuation of previous message
            current_message += " " + line.strip()

    # Add last message
    if current_message:
        messages.append([current_date, current_time, sender, current_message.strip()])

    df = pd.DataFrame(messages, columns=["Date", "Time", "Sender", "Message"])
    return df
