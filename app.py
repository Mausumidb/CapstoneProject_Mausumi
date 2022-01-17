#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request, render_template

import numpy as np
import pandas as pd
import pickle
import nltk
import gzip
#nltk.download('punkt', download_dir='/app/nltk_data/')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',len=0)

@app.route("/predict", methods=['POST'])

def predict():
    if (request.method == 'POST'):
        
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #user_input = request.form["username"]
        #print(user_input)
                
        pickled_tfidf_vectorizer = pickle.load(open('Tfidf_vectorizer.pkl','rb'))
        pkl_model = pickle.load(open('Logistic_Regression_final_model.pkl','rb'))
        #user_final_rating_pickled = pickle.load(open('user_final_rating.pkl','rb')) 
        with gzip.open('user_final_rating_pickled.pkl','rb') as f:
            user_final_rating_pickled=pickle.load(f)
        
        id_name_pickled = pickle.load(open('prod_id_name.pkl','rb')) 
        reviews_data_pickled = pickle.load(open('reviews_data.pkl','rb'))
        
        
        product_recommendations = pd.DataFrame(user_final_rating_pickled.loc[user_input]).reset_index()
        
        product_recommendations.rename(columns={product_recommendations.columns[1]: "user_predicted_sentiment" }, inplace = True)
        product_recommendations = product_recommendations.sort_values(by='user_predicted_sentiment', ascending=False)[0:20]
        #print(product_recommendations)
       
         
    
        product_recommendations.rename(columns={product_recommendations.columns[0]: "prod_name" }, inplace = True)
        id_name_pickled.rename(columns={id_name_pickled.columns[0]: "prod_name" }, inplace = True)  
        reviews_data_pickled.rename(columns={reviews_data_pickled.columns[0]: "prod_name" }, inplace = True)
       
    
        product_recommendations = pd.merge(product_recommendations,id_name_pickled, left_on="prod_name", right_on="prod_name", how = "left")
        
        sentiment_recommendations= pd.merge(product_recommendations,reviews_data_pickled[['prod_name','reviews_text_title']], left_on='prod_name', right_on='prod_name', how = 'left')            
              
             
        user_test_data= pickled_tfidf_vectorizer.transform(sentiment_recommendations['reviews_text_title'].values.astype('U'))
        
        user_sentiment_prediction = pkl_model.predict(user_test_data)
        user_sentiment_prediction = pd.DataFrame(user_sentiment_prediction, columns=['predicted_sentiment'])
        #print(user_sentiment_prediction)

        sentiment_recommendations= pd.concat([sentiment_recommendations, user_sentiment_prediction], axis=1)
       
        
        p=sentiment_recommendations.groupby('prod_name')
        df_predicted_sentiment=pd.DataFrame(p['predicted_sentiment'].count()).reset_index()
        df_predicted_sentiment.columns = ['prod_name', 'Total_review_counts']
        final=pd.DataFrame(p['predicted_sentiment'].sum()).reset_index()
        final.columns = ['prod_name', 'Total_predicted_positive_review_counts']
        
        
        final_sentiment_recommendations=pd.merge(df_predicted_sentiment, final, left_on='prod_name', right_on='prod_name', how='left')
        
        final_sentiment_recommendations['positive_sentiment_percent'] =     final_sentiment_recommendations['Total_predicted_positive_review_counts'].div(final_sentiment_recommendations['Total_review_counts']).replace(np.inf, 0)
        
        
        final_sentiment_recommendations= final_sentiment_recommendations.sort_values(by=['positive_sentiment_percent'], ascending=False )
        final_sentiment_recommendations=pd.merge(final_sentiment_recommendations, id_name_pickled, left_on='prod_name', right_on='prod_name', how='left')
     
        
        top5=final_sentiment_recommendations.head()
        top5=top5['prod_name']
        #print(top5)
        
        
        output=top5.to_list()
        #print(output)
        
        
        
        return render_template('index.html', len=len(output),prediction_text=output)
    else :
        return render_template('index.html',len=0)
    
if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)
