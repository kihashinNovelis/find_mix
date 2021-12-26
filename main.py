import pandas as pd
import numpy as np
import streamlit as st
import pymc3 as pm
import arviz as az
import pickle


with open('comb_data.pkl', 'rb') as f:
    comb_data = pickle.load(f)

origin_list = ['Taiwan','Australia','Japan','Korea','Others','Mexico','Thailand']
mdl_data = comb_data[['UBC','Class1','Class3','Prime','RSI','black2','Total'] + origin_list].dropna()
var_list = ['UBC','Class1','Class3','Prime','RSI']  + origin_list

for var in var_list:
    mdl_data[var] = mdl_data[var]/1000000
mdl_data['black2_ln'] = np.log(mdl_data['black2']/mdl_data['Total'])
mdl_train = mdl_data

with open('trace.pkl', 'rb') as f:
    trace = pickle.load(f)

var_list = ['Class1', 'Class3', 'Prime', 'RSI', 'Taiwan', 'Australia', 'Japan', 'Korea', 'Others', 'Mexico', 'Thailand']
var_dict = {}

def model_factory(mdl_train):
    basic_model = pm.Model()

    with basic_model as bm:

        for var in var_list:
            var_dict[var] = pm.Data(var, mdl_train[var])

        # define priors
        taiwan_p = pm.Normal('taiwan_p', mu=0, sd=1)
        australia_p = pm.Normal('australia_p', mu=0, sd=1)
        japan_p = pm.Normal('japan_p', mu=0, sd=1)
        korea_p = pm.Normal('korea_p', mu=0, sd=1)
        others_p = pm.Normal('others_p', mu=0, sd=1)
        mexico_p = pm.Normal('mexico_p', mu=0, sd=1)
        thailand_p = pm.Normal('thailand_p', mu=0, sd=1)
        ubc_amount = taiwan_p * var_dict['Taiwan'] + australia_p * var_dict['Australia'] \
                     + japan_p * var_dict['Japan'] + korea_p * var_dict['Korea'] + others_p * var_dict['Others'] \
                     + mexico_p * var_dict['Mexico'] + thailand_p * var_dict['Thailand']

        intercept = pm.Normal('intercept', mu=0, sd=1)
        ubc_p = pm.Normal('ubc_p', mu=0.4, sd=1)
        class1_p = pm.Normal('class1_p', mu=0, sd=1)
        class3_p = pm.Normal('class3_p', mu=0, sd=1)
        prime_p = pm.Normal('prime_p', mu=0, sd=1)
        rsi_p = pm.Normal('rsi_p', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)

        mean = intercept + ubc_p * ubc_amount + class1_p * var_dict['Class1'] + class3_p * var_dict['Class3']\
               + prime_p * var_dict['Prime'] + rsi_p * var_dict['RSI']

        # predictions
        observed = pm.Normal('observation', mu=mean, sd=sigma, observed=mdl_train['black2_ln'])

        return bm


if __name__ == '__main__':

    st.title('% of Black Dross Estimation')
    st.text('Use the sidebar on the left to input material weights')
    st.text('Please input material weight(KG) to estimate % black dross')

    min_value = 1
    prime_max_value = int(round(comb_data['Prime'].quantile(0.95),-3))
    class1_max_value = int(round(comb_data['Class1'].quantile(0.95), -3))
    class3_max_value = int(round(comb_data['Class3'].quantile(0.95), -3))
    rsi_max_value = int(round(comb_data['RSI'].quantile(0.95), -3))
    korea_max_value = int(round(comb_data['Korea'].quantile(0.95), -3))
    japan_max_value = int(round(comb_data['Japan'].quantile(0.95), -3))
    thailand_max_value = int(round(comb_data['Thailand'].quantile(0.95), -3))
    australia_max_value = int(round(comb_data['Australia'].quantile(0.95), -3))
    mexico_max_value = int(round(comb_data['Mexico'].quantile(0.95), -3))
    taiwan_max_value = int(round(comb_data['Taiwan'].quantile(0.95), -3))
    others_max_value = int(round(comb_data['Others'].quantile(0.95), -3))


    prime_sidebar = st.sidebar.slider("Prime", min_value, prime_max_value)
    class1_sidebar = st.sidebar.slider("Class 1", min_value, class1_max_value)
    class3_sidebar = st.sidebar.slider("Class 3", min_value, class3_max_value)
    rsi_sidebar = st.sidebar.slider("RSI", min_value, rsi_max_value)

    korea_sidebar = st.sidebar.slider('Korea', min_value, korea_max_value)
    japan_sidebar = st.sidebar.slider('Japan', min_value, japan_max_value)
    thailand_sidebar = st.sidebar.slider('Thailand', min_value, thailand_max_value)
    australia_sidebar = st.sidebar.slider('Australia', min_value, australia_max_value)
    mexico_sidebar = st.sidebar.slider('Mexico', min_value, mexico_max_value)
    taiwan_sidebar = st.sidebar.slider('Taiwan', min_value, taiwan_max_value)
    others_sidebar = st.sidebar.slider('All others', min_value, others_max_value)

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

    ubc_wt =  (taiwan_wt + australia_wt + japan_wt + korea_wt + others_wt + mexico_wt + thailand_wt)
    all_wt = (class1_wt + class3_wt + prime_wt + rsi_wt) + ubc_wt

    s = pd.Series([class1_wt, class3_wt, prime_wt, rsi_wt, taiwan_wt, australia_wt, japan_wt, korea_wt, others_wt, mexico_wt, thailand_wt], index=var_list)
    s = pd.concat([s, s], axis=1).T


    # s = mdl_train.loc['2021-11-04'][var_list]
    # s = pd.concat([s, s], axis=1).T

    if (ubc_wt/all_wt) > 0.7:

        st.write('%UBC: ', '{:.1%}'.format((ubc_wt/all_wt)))

        with model_factory(mdl_train):

            for var in var_list:
                pm.set_data({var: s[var]})

            post_pred = pm.sample_posterior_predictive(samples=1000, trace=trace)['observation']

            print(post_pred.mean())
            prob = np.exp(post_pred.mean())

            lower = np.exp(pd.Series(post_pred.flatten()).quantile(0.05))
            upper = np.exp(pd.Series(post_pred.flatten()).quantile(0.95))

        st.subheader('Estimated % Black Dross:')
        st.subheader('{:.2%}'.format(prob))
        st.write('90% Confidence Interval: ', '{:.2%}'.format(lower), '~', '{:.2%}'.format(upper))

        st.success('Calculation Completed')

    else:
        st.subheader('Model Requires Realistic Inputs!')

