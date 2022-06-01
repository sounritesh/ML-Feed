import streamlit as st
import pandas as pd
import numpy as np
import os
from math import exp

st.title("Personalized Feed Test")

BASE_DIR = "DataFiles/"

st.cache()
def load_data():
    ct = pd.read_csv(os.path.join(BASE_DIR, "child_thoughts_scored.csv"))
    pt = pd.read_csv(os.path.join(BASE_DIR, "final/modeled_thoughts.csv"))
    pt.dropna(inplace=True)
    pt.topic = pt.topic.astype(int)

    ct.created_at = pd.to_datetime(ct.created_at)
    pt.created_at = pd.to_datetime(pt.created_at)

    # print(type(ct.c_parent.iloc[0]), type(pt._id.iloc[0]))

    # print(f"ct len: {len(ct)}")
    tmp_p = pt.rename(columns={"_id": "c_parent"}, errors="raise")
    ct = ct.merge(tmp_p[["c_parent", "topic"]], on="c_parent")
    # print(f"ct len: {len(ct)}")
    # print(ct)

    scores = pd.read_csv(os.path.join(BASE_DIR, "user_scores.csv"))
    scores_arr = np.load(os.path.join(BASE_DIR, "user_scores.npy"))

    topic_map = pd.read_csv(os.path.join(BASE_DIR, "final/topic_map.csv"))
    probs = np.load(os.path.join(BASE_DIR, "final/probabilities.npy"))
    return ct, pt, scores, scores_arr, topic_map, probs

ct, pt, scores, scores_arr, topic_map, probs = load_data()

print((ct.iloc[0].created_at - ct.iloc[1].created_at).seconds/86400)

def decay(x, threshold=0.2, unit='m'):
    # print(x)
    if unit == 'h':
        t = x/3600
    elif unit == 'd':
        t = x/86400
    elif unit == 'w':
        t = x/604800
    elif unit == 'm':
        t = x/2.628e+6

    # print(f"time diff: {t}")
    w = (20-exp(t))/20
    # print(f"weight: {w}")
    if w < threshold:
        return threshold
    else:
        return w


def calculate_user_score(user, threshold=0.2, unit='m'):
    score_pol = [0]*250
    score = [0]*250
    tmp_c = ct[ct["author"] == user]
    for i, row in tmp_c.iterrows():
        p_topic = int(pt[pt["_id"] == row.c_parent].iloc[0].topic)
        last_child = tmp_c[(tmp_c['topic'] == p_topic) & (tmp_c['created_at'] < row.created_at)].sort_values(by='created_at', ascending=False)
        if len(last_child) > 0:
            last_child = last_child.iloc[0]
            t_diff = (row.created_at - last_child.created_at).seconds
        else:
            t_diff = 0

        d = decay(t_diff, threshold, unit)
        score_pol = [s*d for s in score_pol]
        score_pol[p_topic] += row.nli_score

        score = [s*d for s in score]
        score[p_topic] += 1

    # score_main = [i*j for i, j in zip(score, score_pol)]
    # print(score_main)
    return np.array(score), np.array(score_pol)



def get_top_topics(user, n=2, threshold=0.2, unit='m'):
    # ind = scores[scores['user'] == user].index.tolist()[0]
    # score_list = scores_arr[ind]
    score, score_pol = calculate_user_score(user, threshold, unit)

    score = score*topic_map.weight.values

    sorted_inds = np.argsort(score)
    topics_sorted = list(sorted_inds)
    print(score, topics_sorted)
    # print(np.sort(score_list))

    n3 = topics_sorted[-3*n:]
    # top_2n = topics_sorted[-2*n:]
    ###########################################################


    n_pol = np.take(score_pol, n3)

    for i in range(2*n-1):
        for j in range(2*n-i-1):
            if n_pol[j] > n_pol[j+1]:
                t = n3[j]
                n3[j] = n3[j+1]
                n3[j+1] = t

    top_n = n3[-n:]
    bot_n = n3[0:n]

    print(n_pol, n3)

    return top_n, bot_n

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
        n = st.number_input('N', min_value=1, max_value=40, value=2, step=1)
        unit = st.selectbox(
            'Time decay unit',
            ['m', 'w', 'd', 'h'],
            key = 3
        )
        w = st.number_input('Threshold weight', min_value=0.1, max_value=1.0, value=0.2, step=0.05)

if option == "Feed":
    st.subheader("Feed")
    top, bot = get_top_topics(user, n, threshold=w, unit=unit)
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

    past_activities = get_past_activity(user, 200)

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
