import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
with open("faq.json", "r") as f:
    faq = json.load(f)

questions = faq["questions"]
answers = faq["answers"]

# Vectorize questions
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    if similarity[0][idx] > 0.3:  # confidence threshold
        return answers[idx]
    else:
        return "Sorry, I don't understand your question. Please contact support."

# Chat loop
print("🤖 Chatbot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Chatbot: Goodbye! Have a nice day.")
        break
    print("🤖 Chatbot:", chatbot_response(user_input))
