import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline

# Model and tokenizer names (you can choose a different model from Hugging Face)
model_name = "distilbert/distilbert-base-cased-distilled-squad"
tokenizer_name = model_name

access_token = st.secrets["API_key"]

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=access_token)

def app():

  # Load the question-answering pipeline
  question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

  # Create the Streamlit app layout
  st.title("Document Question Answering")

  # Input fields for document and question
  document = st.text_area("Enter the document:")
  question = st.text_input("Enter the question:")

  # Button to trigger the answer generation
  if st.button("Answer Question"):
      # Interact with the pipeline model
      result = question_answerer(question=question, context=document)

      # Display the answer
      st.write("Answer:", result["answer"])
      st.write("Score:", result["score"])  # Optionally display the confidence score


#run the app
if __name__ == "__main__":
  app()
