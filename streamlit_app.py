import streamlit as st
import pandas as pd
import numpy as np
import os

st.title("Personalized Feed Test")

BASE_DIR = "DataFiles/"

st.cache()
def load_data():
    ct = pd.read_csv(os.path.join(BASE_DIR, "child_thoughts_scored.csv"))
    pt = pd.read_csv(os.path.join(BASE_DIR, "modeled_thoughts_160.csv"))
    pt.dropna(inplace=True)
    pt.topic = pt.topic.astype(int)

    scores = pd.read_csv(os.path.join(BASE_DIR, "user_scores.csv"))
    scores_arr = np.load(os.path.join(BASE_DIR, "user_scores.npy"))

    topic_map = pd.read_csv(os.path.join(BASE_DIR, "topic_map.csv"))
    return ct, pt, scores, scores_arr, topic_map

ct, pt, scores, scores_arr, topic_map = load_data()

def get_top_topics(user, n=2):
    ind = scores[scores['user'] == user].index.tolist()[0]
    score_list = scores_arr[ind]
    sorted_inds = np.argsort(score_list)
    topics_sorted = list(sorted_inds)

    top_5 = topics_sorted[0:n]
    bot_5 = topics_sorted[-n:]

    return top_5, bot_5

def get_feed_thoughts(user, top, bot, s=20):
    tmp_c = ct[ct["author"] == user]
    print(len(tmp_c))

    tmp_top = pt.loc[pt.topic.isin(top), :]
    tmp_bot = pt.loc[pt.topic.isin(bot), :]
    # tmp = pt

    tmp_top = tmp_top[tmp_top["author"] != user]
    tmp_top = tmp_top.loc[~tmp_top["_id"].isin(tmp_c.c_parent), :]

    tmp_bot = tmp_bot[tmp_bot["author"] != user]
    tmp_bot = tmp_bot.loc[~tmp_bot["_id"].isin(tmp_c.c_parent), :]

    # tmp = tmp[tmp["author"] != user]
    # tmp = tmp.loc[~tmp["_id"].isin(tmp_c.c_parent), :]

    tmp_top.created_at = pd.to_datetime(tmp_top.created_at)
    tmp_bot.created_at = pd.to_datetime(tmp_bot.created_at)
    # tmp.created_at = pd.to_datetime(tmp.created_at)
    return pd.concat(
        [
            # tmp.sort_values(by="created_at", ascending=False).iloc[0:s],
            tmp_top.sort_values(by="created_at", ascending=False).iloc[0:s],
            tmp_bot.sort_values(by="created_at", ascending=False).iloc[0:s]
        ]
    ).sort_values(by='created_at', ascending=False)

def get_past_activity(user, m=50):
    tmp = ct[ct["author"] == user]
    tmp.created_at = pd.to_datetime(tmp.created_at)
    try:
        tmp = tmp.sort_values(by="created_at", ascending=False).iloc[:m]
    except:
        tmp = tmp.sort_values(by="created_at", ascending=False)

    parent_list = []
    child_list = []
    date_list = []
    topic_list = []
    for i, row in tmp.iterrows():
        p = pt[pt['_id'] == row.c_parent].iloc[0]
        parent_list.append(p.body)
        child_list.append(row.body)
        date_list.append(row.created_at)
        topic_list.append(topic_map.iloc[p.topic].topic)

    return zip(parent_list, child_list, date_list, topic_list)

with st.sidebar:
    option = st.selectbox(
        'What to look at?',
        ['Feed', 'Previous Activity'],
        key = 1
    )

    user = st.selectbox(
        'Select User',
        scores.user.tolist(),
        key = 2
    )

if option == "Feed":
    st.subheader("Feed")
    top, bot = get_top_topics(user)
    st.write("{}'s shortlisted topics: {}, {}".format(user, top, bot))

    feed_thoughts = get_feed_thoughts(user, top, bot)

    for i, row in feed_thoughts.iloc[:50].iterrows():
        st.markdown(
            '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous"> \
            <div class="card text-light bg-secondary mb-3"> \
        <div class="card-header">{} --- {}</div> \
        <div class="card-body"> \
            <p class="card-text text-light">{}</p> \
        </div> \
        </div>'.format(row.created_at, topic_map.iloc[row.topic].topic, row.body), unsafe_allow_html=True
        )
else:
    st.subheader("Past Activity")

    past_activities = get_past_activity(user)

    for parent, child, date, topic in past_activities:
        st.markdown(
                '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous"> \
                <div class="card text-light bg-secondary mb-3"> \
            <div class="card-header">{} --- {}</div> \
            <div class="card-body"> \
                <p class="card-text text-dark">{}</p> \
                <p class="card-text text-light">{}</p> \
            </div> \
            </div>'.format(date, topic, parent, child), unsafe_allow_html=True
            )
