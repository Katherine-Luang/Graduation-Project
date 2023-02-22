import nltk
import streamlit as st
import pandas as pd
import os
import wordcloud
import re
import spacy_streamlit
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from transformers import AutoTokenizer, AutoModelWithLMHead
from annotated_text import annotated_text
from PIL import Image
from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP
from nltk.util import ngrams
from collections import Counter
import sqlite3
import spacy

connection = sqlite3.connect("corpora_data.db")

st.set_page_config(
    page_title="a Python Tool for Visualizing and Analyzing Domain-Specific Corpora",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_df_from_db(sql):
    cursor = connection.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    columnDes = cursor.description  # 获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df = pd.DataFrame([list(i) for i in data], columns=columnNames)
    # print(df)
    return df


def intro():
    st.write("### A Python Tool for Visualizing and Analyzing Domain-Specific Corpora 👋:)")
    st.sidebar.success("Select a level above.")

    st.markdown(
        """
        
This is a tool for comparative analyses of domain-specific corpora at the level of word, collocation and sentence, using Python, Pandas(a data analysis Python library), Plotly(a visualization Python library) and Streamlit(an open-source Python library).

This software, which analyses the range of vocabulary in texts at different levels, provides the type, the token and TTR(the type-toke ratio), a word frequency figure (the total number of times the word appears in all the texts), parts of speech proportion for each word in the text, the word lengths of each word; the entities occur in a sentence, the dependency parse of the sentence, the word cloud of each corpus. 

It can be used to compare the linguistic features of different corpora to have a better understanding of language characteristics. The program is free for everyone to use and is downloadable from https://gitee.com/Edward-Wu/graduation-project) and certain packages need to be downloaded or some functions might not work well. Please follow the instruction in the file readme.md.

### Our website features

1️⃣	A corpus of over 80 authoritative textbooks in 8 domains (including business, economics, history, linguistics, management, media communication, philosophy and psychology)

2️⃣	A customized combination of texts for a contrastive analysis 

3️⃣	A comprehensive analysis of the selected texts at the level of word, collocation and sentence       
    """
    )


def Traverse_Path(Path):
    """
        yzuy
        遍历路径，获取路径下的文件列表
        :param Path: 目标路径
        :return: FileList: 路径下的文件列表
    """
    FileList = []
    for roots, dirs, files in os.walk(Path):
        for file in files:
            FilePath = os.path.join(roots, file)
            FileList.append(FilePath)
    return FileList


def overview():
    st.title("Overview")

    fields = (
        'General', 'Business', 'Economics', 'History', 'Linguistics', 'Management', 'Media_communication', 'Philosophy',
        'Psychology')

    Features = (
        "Wiki Knowledge Features",
        "Entity Density Features",
        "Phrasal Features",
        "Tree Features",
        "Part-of-Speech Features",
        "TTR Features",
        "Psycholinguistic Difficulty",
        "Shallow Features",
        "Traditional Formulas"
    )

    Features_meaning = {
        "Wiki Knowledge Features": "WoKF",
        "Entity Density Features": "EnDF",
        "Phrasal Features": "PhrF",
        "Tree Features": "TrSF",
        "Part-of-Speech Features": "POSF",
        "TTR Features": "TTRF",
        "Psycholinguistic Difficulty": "PsyF",
        "Shallow Features": "ShaF",
        "Traditional Formulas": "TraF",
        "General": "BNC_Baby"
    }

    books = dict({})  # 所有书名的字典

    OriginPath = '.\\book'  # 待测文本存储位置
    FileList = Traverse_Path(OriginPath)
    for field in fields:
        for file in FileList:
            if field in file:
                if field not in books:
                    books[field] = []
                else:
                    books[field].append(file)

    option1 = st.multiselect(
        'Please select fields',
        fields)

    option2 = st.selectbox(
        'Please select features',
        Features
    )

    df = get_df_from_db(f"select * from {Features_meaning[option2]}")
    df.set_index('field1', inplace=True)
    df = df.T
    index_name = df.index.values
    dict_index = dict(zip(df.Definition, index_name))

    option3 = st.selectbox(
        'Please select a feature',
        df.Definition
    )

    final_df = df[option1][df.index.str.contains(f'{dict_index[option3]}')].T
    final_df[f"{dict_index[option3]}"] = final_df[f"{dict_index[option3]}"].astype(float)

    st.bar_chart(data=final_df)

    # option1.append("Definition")
    # st.table([option1])


def word():
    st.title("Word-level Analysis")

    options = st.multiselect('Select 2~3 fields that you would like to compare',
                             ['General', 'Business', 'Economics', 'History', 'Linguistics', 'Management',
                              'Media_communication',
                              'Philosophy', 'Psychology'],
                             ['General', 'Business', 'History'],
                             max_selections=3)

    # 1.Basic Information
    st.subheader("Basic Information")

    # 可视化basic information：type，token，TTR
    basic_info = pd.read_excel(rf".\word_attribute\basic_information.xlsx", index_col=0)
    df_basic_info = pd.DataFrame(basic_info)  # basic information全部数据转化成dataframe格式
    # 选择需要显示的数据
    df = df_basic_info.loc[options]
    st.dataframe(df, use_container_width=True)

    # 显示bar chart
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Type, Token and TTR of Each Corpus", "Type of Each Corpus", "Token of Each Corpus", "TTR of Each Corpus"])
    with tab1:
        fig = px.bar(df.T,
                     barmode='group',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     title="Type, Token and TTR of Each Corpus",
                     labels={"variable": "Field"},
                     height=500)

        fig.update_xaxes(title_text=None)
        fig.update_yaxes(title_text='Number')
        fig.update_traces(hovertemplate='%{x}<br>Number : %{y:,.}')

        st.plotly_chart(fig, theme=None, use_container_width=True)

    with tab2:
        Type = df.T.loc['Type']
        fig = px.bar(Type,
                     barmode='group',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     title="Type of Each Corpus",
                     labels={"variable": ""},
                     height=400)

        fig.update_xaxes(title_text='Field')
        fig.update_yaxes(title_text='Number')
        fig.update_traces(hovertemplate='%{x}<br>Number : %{y:,.}')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    with tab3:
        Token = df.T.loc['Token']
        fig = px.bar(Token,
                     barmode='group',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     title="Token of Each Corpus",
                     labels={"variable": ""},
                     height=400)

        fig.update_xaxes(title_text='Field')
        fig.update_yaxes(title_text='Number')
        fig.update_traces(hovertemplate='%{x}<br>Number : %{y:,.}')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    with tab4:
        TTR = df.T.loc['TTR']
        fig = px.bar(TTR,
                     barmode='group',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     title="TTR of Each Corpus",
                     labels={"variable": ""},
                     height=400)

        fig.update_xaxes(title_text='Field')
        fig.update_yaxes(title_text='Value')
        fig.update_traces(hovertemplate='%{x}<br>Value : %{y:,.}')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    # 2.Information about the Search Word
    def search_word_freq_and_pos(search_word, fields):
        df = pd.DataFrame()
        for field in fields:
            word = pd.read_excel(rf".\word_attribute_pos_count\{field}_word_attribute.xlsx")
            word_df = pd.DataFrame(word)
            new_df = word_df.query("word == @search_word")
            if (len(new_df) != 0):
                new_df.loc[:, 'field'] = field
                df = pd.concat([df, new_df])
        if len(df) != 0:
            data_load_state = st.info('Loading ...', icon="🤔")
            fig = px.bar(df,
                         x="count",
                         y="field",
                         color="pos_tag",
                         color_discrete_sequence=px.colors.diverging.Portland,
                         title=f'Word Frequency and POS Proportion of "{search_word}"',
                         labels={'pos_tag': 'Part of Speech'},
                         text_auto=True,
                         orientation='h',
                         height=550)

            fig.update_xaxes(title_text='Count')  # ,showticklabels=False
            fig.update_yaxes(title_text='Field', dtick=1)
            st.plotly_chart(fig, theme=None, use_container_width=True)
            data_load_state.success('Loading graphs...done!', icon="😊")
        else:
            st.error(f'{search_word}  is not in the corpora. Please try another word!', icon="🚨")

    st.subheader("Information about the Search Word")
    search_word = st.text_input('Search a word', 'trapped')

    if len(options) == 0:
        st.warning('Please select at least one field!', icon="⚠️")
    else:
        if search_word:
            search_word_freq_and_pos(search_word, options)

    # 3.Word Frequency
    st.subheader("Word Frequency & Part of Speech")

    # 用于显示word frequency水平条形统计图
    def word_freq_chart(field, number):
        word_freq = pd.read_excel(rf".\word_freq\{field}_word_frequencies.xlsx")
        word_freq_df = pd.DataFrame(word_freq)
        word_freq_showed = word_freq_df.sort_values(by="freq", ascending=False).head(int(number))
        # print(word_freq_showed)
        word_freq_showed = word_freq_showed.sort_values(by='freq', ascending=True)
        fig = px.bar(word_freq_showed,
                     x="freq",
                     y="word",
                     color_discrete_sequence=px.colors.sequential.Cividis,
                     title=rf"The First {number} Words of {field} Corpus",
                     text_auto=True,
                     orientation='h',
                     height=600)
        fig.update_xaxes(title_text='Number')
        fig.update_yaxes(title_text='Word')
        fig.update_traces(hovertemplate='Word : %{y:,.}<br>Number : %{x}')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    # 读取word frequency数据
    def read_word_freq(field):
        word_freq = pd.read_excel(rf".\word_freq\{field}_word_frequencies.xlsx")
        word_freq_df = pd.DataFrame(word_freq)
        df = word_freq_df.sort_values(by="freq", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        st.dataframe(df, use_container_width=True)

    # 生成pos proportion pie chart
    def pos_proportion_pie_chart(field):
        pos_proportion = pd.read_excel(rf".\pos_proportion\{field}_pos_proportion.xlsx")
        pos_proportion_df = pd.DataFrame(pos_proportion)
        fig = px.pie(pos_proportion_df,
                     values='percentage(%)',
                     names='pos_tag',
                     title=f"Part of Speech Proportion of {field} Corpus",
                     color_discrete_sequence=px.colors.sequential.Agsunset)

        st.plotly_chart(fig, theme=None, use_container_width=True)

    def pos_proportion_bar_chart(field):
        pos_proportion = pd.read_excel(rf".\pos_proportion\{field}_pos_proportion.xlsx")
        pos_proportion_df = pd.DataFrame(pos_proportion)
        fig = px.bar(pos_proportion_df,
                     x="pos_tag",
                     y="percentage(%)",
                     color_discrete_sequence=px.colors.sequential.Agsunset,
                     title=f"Part of Speech Proportion of {field} Corpus",
                     height=500)

        fig.update_xaxes(title_text='Part of Speech')
        fig.update_yaxes(title_text='Percentage', ticksuffix="%")
        fig.update_traces(hovertemplate='%{x}<br>Percentage : %{y:,.4f}%')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    def show_first_few_word_by_pos(field, pos, count):
        word_pos = pd.read_excel(rf".\word_attribute_pos_count\{field}_word_attribute.xlsx")
        word_pos_df = pd.DataFrame(word_pos)
        word_pos_df = word_pos_df.query('pos_tag==@pos').head(int(count))
        word_pos_df = word_pos_df.sort_values(by='count', ascending=True)

        fig = px.bar(word_pos_df,
                     x="count",
                     y="word",
                     color_discrete_sequence=px.colors.diverging.Fall,
                     title=rf"The First {count} Words in {pos} of {field} Corpus",
                     text_auto=True,
                     orientation='h')

        fig.update_xaxes(title_text='Count')  # ,showticklabels=False
        fig.update_yaxes(title_text='Word', dtick=1)
        fig.update_traces(hovertemplate='Word : %{y}<br>Count : %{x:,.}')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    # 累积频率图
    def cumulative_frequency_graph(field, number):
        cumul_freq = pd.read_excel(rf".\cumulative_word_frequency\{field}_cumulative_word_frequency.xlsx")
        cumul_freq_df = pd.DataFrame(cumul_freq).head(int(number))

        if number <= 50:
            fig = px.line(cumul_freq_df,
                          x='word',
                          y='cumulative_freq',
                          title=f'Cumulative Word Frequency of the First {number} Common Words',
                          color_discrete_sequence=px.colors.sequential.haline,
                          height=500)

            fig.update_xaxes(title_text='Common Words')
            fig.update_yaxes(title_text='Cumulative Counts')
            fig.update_traces(hovertemplate='Common Word : %{x}<br>Cumulative Counts : %{y:,.}')
            st.plotly_chart(fig, theme=None, use_container_width=True)
        else:
            arr = [x for x in range(1, int(number) + 1)]
            fig = px.line(cumul_freq_df,
                          x=arr,
                          y='cumulative_freq',
                          title=f'Cumulative Word Frequency of the First {number} Common Words',
                          color_discrete_sequence=px.colors.sequential.haline,
                          height=500)
            if number <= 100:
                tick = 10
            elif 100 < number < 500:
                tick = 50
            else:
                tick = 100
            fig.update_xaxes(title_text='Common Words', dtick=tick)
            fig.update_yaxes(title_text='Cumulative Counts')
            fig.update_traces(hovertemplate='Common Word : %{x}<br>Cumulative Counts : %{y:,.}')
            st.plotly_chart(fig, theme=None, use_container_width=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Word Frequency Graphs",
                                                  "Word Frequency Lists",
                                                  "Cumulative Word Frequency Graphs",
                                                  "Part of Speech Proportion Pie Chart",
                                                  "Part of Speech Proportion Bar Chart",
                                                  "Word Frequency Displayed Through Part of Speech",
                                                  ])
    with tab1:
        number_showed = st.slider(
            'Select the number of the first few words displayed',
            5, 100, 10)

        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading graphs...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    word_freq_chart(options[0], number_showed)
            with col2:
                if len(options) > 1:
                    word_freq_chart(options[1], number_showed)
            with col3:
                if len(options) > 2:
                    word_freq_chart(options[2], number_showed)
            data_load_state.success('Loading graphs...done!', icon="😊")
    with tab2:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading data...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    st.caption(options[0])
                    read_word_freq(options[0])
            with col2:
                if len(options) > 1:
                    st.caption(options[1])
                    read_word_freq(options[1])
            with col3:
                if len(options) > 2:
                    st.caption(options[2])
                    read_word_freq(options[2])
            data_load_state.success('Loading data...done!', icon="😊")

    with tab3:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            number = st.number_input('Input a number (10~5000)', 10, 5000, 100, 10)

            data_load_state = st.info('Loading graphs...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    cumulative_frequency_graph(options[0], number)

            with col2:
                if len(options) > 1:
                    cumulative_frequency_graph(options[1], number)

            with col3:
                if len(options) > 2:
                    cumulative_frequency_graph(options[2], number)
            data_load_state.success('Loading graphs...done!', icon="😊")

    with tab4:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading graphs...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    pos_proportion_pie_chart(options[0])
            with col2:
                if len(options) > 1:
                    pos_proportion_pie_chart(options[1])
            with col3:
                if len(options) > 2:
                    pos_proportion_pie_chart(options[2])
            data_load_state.success('Loading graphs...done!', icon="😊")
    with tab5:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading graphs...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    pos_proportion_bar_chart(options[0])
            with col2:
                if len(options) > 1:
                    pos_proportion_bar_chart(options[1])
            with col3:
                if len(options) > 2:
                    pos_proportion_bar_chart(options[2])
            data_load_state.success('Loading graphs...done!', icon="😊")

    with tab6:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            option = st.selectbox(
                'Select a  part of speech',
                ['NN', 'FW', 'CD', 'NNPS', 'LS', 'NNP', 'JJ', 'JJS', 'VBZ', 'RB', 'VBD', 'MD', 'IN', 'PRP$', 'JJR',
                 'UH', 'PDT', 'WP$', 'WRB', 'RBR', 'DT', 'POS', 'RP', 'EX', 'NNS', 'VBG', 'WP', 'TO', 'VB', 'RBS',
                 'PRP', 'WDT', 'CC', 'VBN', 'VBP', 'SYM'])

            number = st.slider(
                'Select the number of the first few words displayed',
                10, 100, 10)

            data_load_state = st.info('Loading graphs...', icon="🤔")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(options) > 0:
                    show_first_few_word_by_pos(options[0], option, number)
            with col2:
                if len(options) > 1:
                    show_first_few_word_by_pos(options[1], option, number)
            with col3:
                if len(options) > 2:
                    show_first_few_word_by_pos(options[2], option, number)
            data_load_state.success('Loading graphs...done!', icon="😊")

    # 4.Wordcloud
    st.subheader("Word Cloud")

    def wordcloud_show(field):
        image = Image.open(fr'.\wordcloud\{field}.png')
        st.image(image, caption=f'{field}')

    if len(options) == 0:
        st.warning('Please select at least one field!', icon="⚠️")
    else:
        data_load_state = st.info('Loading graphs...', icon="🤔")
        col1, col2, col3 = st.columns(3)
        with col1:
            if len(options) > 0:
                wordcloud_show(options[0])
        with col2:
            if len(options) > 1:
                wordcloud_show(options[1])
        with col3:
            if len(options) > 2:
                wordcloud_show(options[2])
        data_load_state.success('Loading graphs...done!', icon="😊")

    # 5.Word Lengths
    def word_lengths_freq_bar_chart(fields):
        fig = go.Figure()
        colors = ['#A56CC1', '#A6ACEC', '#63F5EF']
        for field in fields:
            word_length = pd.read_excel(rf".\word_freq\{field}_word_frequencies.xlsx")
            word_length_df = pd.DataFrame(word_length)
            word_length_df.groupby(['word_lengths']).size()
            df = pd.DataFrame(word_length_df.groupby(['word_lengths']).size()).reset_index()
            df.columns = ["word_lengths", "count"]
            df.loc[:, "field"] = field

            fig.add_trace(go.Bar(x=df["word_lengths"],
                                 y=df["count"],
                                 name=f'{field}'
                                 ))

        fig.update_xaxes(title_text='Word Lengths', dtick=1)
        fig.update_yaxes(title_text='Count')
        fig.update_layout(barmode='group',
                          height=550,
                          bargap=0.05,
                          bargroupgap=0.1,
                          title_text='Word Lengths Bar Chart',
                          colorway=colors)
        st.plotly_chart(fig, theme=None, use_container_width=True)

    # 词长直方图
    colors = ['#333F44', '#37AA9C', '#94F3E4']

    def word_lengths_histograms(fields):
        fig = go.Figure()
        for field in fields:
            word_length = pd.read_excel(rf".\word_freq\{field}_word_frequencies.xlsx")
            word_length_df = pd.DataFrame(word_length)
            hist = go.Histogram(x=word_length_df["word_lengths"],
                                name=f'{field}',
                                histnorm='percent')
            fig.add_trace(hist)
        fig.update_layout(barmode='relative',
                          bargap=0.05,
                          title_text='Word Lengths Histograms',
                          xaxis_title_text='Word Lengths',
                          yaxis_title_text='Percentage',
                          height=500,
                          colorway=colors)  # obarmode = "overlay"
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, theme=None, use_container_width=True)

    st.subheader("Word Lengths")
    tab1, tab2 = st.tabs(["Word Lengths Graphs",
                          "Word Lengths Histograms", ])

    with tab1:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading graphs...', icon="🤔")
            word_lengths_freq_bar_chart(options)
            data_load_state.success('Loading graphs...done!', icon="😊")

    with tab2:
        if len(options) == 0:
            st.warning('Please select at least one field!', icon="⚠️")
        else:
            data_load_state = st.info('Loading graphs...', icon="🤔")
            word_lengths_histograms(options)
            data_load_state.success('Loading graphs...done!', icon="😊")

    # 7."The Full Name of Part of Speech"
    st.subheader("The Full Name of Part of Speech")
    full_name = pd.read_excel(r".\word_attribute_pos_count\pos_full_name.xlsx")
    full_name_df = pd.DataFrame(full_name)
    full_name_df.columns = ["abbreviation", "full_name"]
    full_name_df.index = full_name_df.index + 1
    st.dataframe(full_name_df, use_container_width=True)


def sentence():
    st.title("Sentence-level Analysis")

    fields = (
        'Business', 'Economics', 'History', 'Linguistics', 'Management', 'Media_communication', 'Philosophy',
        'Psychology')

    books = dict({})  # 所有书名的字典

    OriginPath = '.\\book'  # 待测文本存储位置
    FileList = Traverse_Path(OriginPath)
    for field in fields:
        for file in FileList:
            if field in file:
                if field not in books:
                    books[field] = []
                else:
                    filename = file.split('\\')[-1]
                    books[field].append(filename[:-4])

    option1 = st.selectbox(
        'Please select a field',
        fields)

    options = st.multiselect(
        'Please select a book',
        books[f'{option1}'],
        [books[f'{option1}'][1]])

    excel_total = pd.ExcelFile(rf".\sentences_attribute\{option1}_sentences_attribute.xlsx")

    if not options:
        st.write("Please select a book!")
    else:
        data_load_state = st.info('Loading data...', icon="🤔")
        st.subheader(f"Basic Information of {option1}")

        basicinfo = pd.read_excel(rf".\sentences_attribute\sentences_total_attribute.xlsx")
        st.dataframe(basicinfo.query('field == @option1'))

        df_all_sen = pd.DataFrame()
        for option in options:
            cut = option[:20]
            for sheet_name in excel_total.sheet_names:
                if cut in sheet_name:
                    df_sen = pd.read_excel(rf'.\sentences_attribute\{option1}_sentences_attribute.xlsx',
                                           sheet_name=sheet_name)
                    df_all_sen = pd.concat([df_all_sen, df_sen])

        sen = st.text_input('Search', 'word')
        st.warning('Double click to see the whole sentence', icon="⚠️")
        df_after = df_all_sen[df_all_sen.sentences.str.contains(f"{sen}")].reset_index().drop('index', axis=1)
        st.dataframe(df_after)

        data_load_state.success('Loading data...done!', icon="😊")

        number = st.number_input('Please type in the index of a sentence for more information', step=1, value=-1)

        nlp = StanfordCoreNLP(r'.\stanfordnlp', lang='en')

        if number >= 0:
            data_load_state = st.info('Loading data...', icon="🤔")
            sentence_tree = df_after.iloc[number].at['sentences']
            data_load_state.success('Loading... done!', icon="😊")

            sen_pos = df_after.iloc[number].at['tagged']
            reg = re.findall(r"\(.+?\)", sen_pos)

            print(reg)
            # 选中句子的信息的分析
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(sentence_tree)
            spacy_streamlit.visualize_parser(doc)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
            spacy_streamlit.visualize_tokens(doc)

        st.balloons()


def Collocation():
    st.header("Collocation")

    word = st.text_input('Search', '', placeholder="Enter a word to see the collocations")
    phrase_length = st.number_input('Phrase Length', min_value=2, max_value=10)
    top = st.number_input('Show Top', min_value=1, value=20)

    col1, col2 = st.columns(2)

    with col1:
        fields = (
            'General', 'Business', 'Economics', 'History', 'Linguistics', 'Management', 'Media_communication',
            'Philosophy',
            'Psychology')

        option1 = st.selectbox(
            'Field A',
            fields)

        n_grams = 'n_grams_' + option1

        if word:
            if word.isalpha():
                data_load_state = st.info('Loading data...', icon="🤔")
                df = get_df_from_db(f"select * from {n_grams} WHERE field3={phrase_length}")
                del df['field3']

                # 删除首行，首行为text
                df = df.iloc[1:, :]
                df.columns = ['Collocations', 'Frequency']
                df = df[df.Collocations.str.lower().str.contains(word.lower(), na=False)].iloc[:top]

                # 将Frequency转化为int
                convert_dict = {'Frequency': int}
                df = df.astype(convert_dict)
                df = df.sort_values('Frequency', ascending=True)

                if len(df) > 0:

                    fig = px.bar(df,
                                 x="Frequency",
                                 y="Collocations",
                                 title="Chart of Field A",
                                 text_auto=True,
                                 orientation='h',
                                 height=600,
                                 )
                    fig.update_xaxes(title_text='Number')
                    fig.update_yaxes(title_text='Word')
                    fig.update_traces(hovertemplate='Word : %{y:,.}<br>Number : %{x}')
                    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, theme=None, use_container_width=True)

                    # st.dataframe(df, width=1000)
                    data_load_state.success('Loading data...done!', icon="😊")

                else:
                    data_load_state.error("No collocations found. Try another word!", icon="😮")
            else:
                st.warning("Your search should contain only letters!", icon="⚠")

    with col2:
        fields = (
            'Economics', 'General', 'History', 'Linguistics', 'Management', 'Media_communication', 'Philosophy',
            'Psychology', 'Business')

        option2 = st.selectbox(
            'Field B',
            fields)

        n_grams = 'n_grams_' + option2

        if word:
            if word.isalpha():
                data_load_state = st.info('Loading data...', icon="🤔")
                df = get_df_from_db(f"select * from {n_grams} WHERE field3={phrase_length}")
                del df['field3']

                # 删除首行，首行为text
                df = df.iloc[1:, :]
                df.columns = ['Collocations', 'Frequency']
                df = df[df.Collocations.str.lower().str.contains(word.lower(), na=False)].iloc[:top]

                # 将Frequency转化为int
                convert_dict = {'Frequency': int}
                df = df.astype(convert_dict)
                df = df.sort_values('Frequency', ascending=True)

                if len(df) > 0:

                    fig = px.bar(df,
                                 x="Frequency",
                                 y="Collocations",
                                 title="Chart of Field B",
                                 text_auto=True,
                                 # orientation='h',
                                 height=600)
                    fig.update_xaxes(title_text='Number')
                    fig.update_yaxes(title_text='Word')
                    fig.update_traces(hovertemplate='Word : %{y:,.}<br>Number : %{x}')
                    st.plotly_chart(fig, theme=None, use_container_width=True)

                    # st.dataframe(df, width=1000)
                    data_load_state.success('Loading data...done!', icon="😊")

                else:
                    data_load_state.error("No collocations found. Try another word!", icon="😮")
            else:
                st.warning("Your search should contain only letters!", icon="⚠")

    st.balloons()


page_names_to_funcs = {
    "Introduction": intro,
    "Overview": overview,
    "Word-level Analysis": word,
    "Collocation": Collocation,
    "Sentence-level Analysis": sentence,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
