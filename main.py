import pandas as pd
import numpy as np
import streamlit as st

trace_stats_df = pd.read_pickle('trace_stats_df.pkl')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    st.title('% of Black Dross Estimation')
    st.text('Use the sidebar on the left to input material weights')
    st.text('Please input material weight(KG) to estimate % black dross')

    min_value = 1
    max_value = 300000

    prime_sidebar = st.sidebar.slider("Prime", min_value, max_value)
    class1_sidebar = st.sidebar.slider("Class 1", min_value, max_value)
    class3_sidebar = st.sidebar.slider("Class 3", min_value, max_value)
    rsi_sidebar = st.sidebar.slider("RSI", min_value, max_value)

    korea_sidebar = st.sidebar.slider('Korea', min_value, max_value)
    japan_sidebar = st.sidebar.slider('Japan', min_value, max_value)
    thailand_sidebar = st.sidebar.slider('Thailand', min_value, max_value)
    australia_sidebar = st.sidebar.slider('Australia', min_value, max_value)
    mexico_sidebar = st.sidebar.slider('Mexico', min_value, max_value)
    taiwan_sidebar = st.sidebar.slider('Taiwan', min_value, max_value)
    others_sidebar = st.sidebar.slider('All others', min_value, max_value)

    prime_wt = st.number_input("Prime", prime_sidebar)/1000000
    class1_wt = st.number_input("Class 1", class1_sidebar)/1000000
    class3_wt = st.number_input("Class 3", class3_sidebar)/1000000
    rsi_wt = st.number_input("RSI", rsi_sidebar)/1000000

    korea_wt = st.number_input('Korea', korea_sidebar)/1000000
    japan_wt = st.number_input('Japan',japan_sidebar)/1000000
    thailand_wt = st.number_input('Thailand',  thailand_sidebar)/1000000
    australia_wt = st.number_input('Australia', australia_sidebar)/1000000
    mexico_wt = st.number_input('Mexico', mexico_sidebar)/1000000
    taiwan_wt = st.number_input('Taiwan',  taiwan_sidebar)/1000000
    others_wt = st.number_input('All others',others_sidebar)/1000000

    all_wt = (prime_wt + class1_wt + class3_wt + rsi_wt + korea_wt +  japan_wt + thailand_wt + australia_wt + mexico_wt + taiwan_wt + others_wt)
    ubc_wt = (korea_wt +  japan_wt + thailand_wt + australia_wt + mexico_wt + taiwan_wt + others_wt)

    if ubc_wt/all_wt > 0.7:

        input_index = ['prime','class1','class3','rsi','korea','japan','thailand','australia','mexico','taiwan','other']
        country_p = trace_stats_df.loc[['korea_p', 'japan_p', 'thailand_p', 'australia_p', 'mexico_p', 'taiwan_p', 'others_p']]*trace_stats_df.loc['ubc_p']
        input_p = trace_stats_df.loc[['prime_p', 'class1_p', 'class3_p', 'rsi_p']]

        coef = pd.Series(np.append(input_p, country_p), index=input_index)
        wt = pd.Series([prime_wt, class1_wt, class3_wt, rsi_wt, korea_wt, japan_wt, thailand_wt, australia_wt, mexico_wt, taiwan_wt, others_wt], index=input_index)
        new_coef = coef.multiply(wt)
        intercept = pd.Series([-3.55421], index=['intercept'])

        z = pd.concat([intercept, new_coef], axis=0)
        prob = np.exp(z).product()

        st.subheader('Estimated % Black Dross:')
        st.subheader('{:.2%}'.format(prob))

    else:
        st.subheader('Model Requires Realistic Inputs!')

