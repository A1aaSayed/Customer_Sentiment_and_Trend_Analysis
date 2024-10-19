import streamlit as st  
import pandas as pd
import altair as alt
import joblib

# Load your trained logistic regression model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# Function to preprocess text (update as needed)
def preprocess_text(text):
    return vectorizer.transform([text])

# Function to predict sentiment using your model
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    return prediction[0]

# Function to analyze token sentiment
def analyze_token_sentiment(docx):
    tokens = docx.split()
    token_sentiments = []
    
    for token in tokens:
        processed_token = preprocess_text(token)
        score = model.predict(processed_token)[0]  # Get prediction
        token_sentiments.append({'token': token, 'score': 'Positive' if score == 1 else 'Negative'})
    
    return pd.DataFrame(token_sentiments)

# Main function
def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            if not raw_text.strip():
                st.error("Please enter some text for analysis.")
                return

            with col1:
                st.info("Overall Sentiment")
                sentiment = predict_sentiment(raw_text)
                st.write(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")

            with col2:
                st.info("Token Sentiment")
                token_sentiments_df = analyze_token_sentiment(raw_text)
                st.dataframe(token_sentiments_df)

    else:
        st.subheader("About")
        st.write("This app performs sentiment analysis using a logistic regression model.")

if __name__ == '__main__':
    main()