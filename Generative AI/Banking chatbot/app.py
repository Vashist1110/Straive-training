from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

faqs = [
    "How can I reset my online banking password?",
    "How do I check my account balance",
    "What should I do if my debit card is lost?",
    "How can I apply for a Personal loan?",
    "How do I activate international transactions on my credit card?"
]

answers = {
    faqs[0]: "You can reset your password by clicking 'Forgot password' on the login page and following the verification steps.",
    faqs[1]: "You can check your balance using the mobile app, internet banking, or by visiting an ATM",
    faqs[2]: "Report the lost debit card immediately through the customer service helpline or the banking app.",
    faqs[3]: "You can apply for a personal loan online through the bank's portal or by visiting your nearest branch.",
    faqs[4]: "International transaction can be activated via the mobile banking app or by contacting customer care."
}

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_embeddings = model.encode(faqs)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(faq_embeddings))

user_query = "How do I reset my online banking password?"
query_embedding = model.encode([user_query])

number_of_result = 1
dist, ind = index.search(np.array(query_embedding), number_of_result)

first_match_index = ind[0][0]  # ✅ Fixed variable name here
matched_faq = faqs[first_match_index]
matched_answer = answers[matched_faq]

print("User Question:", user_query)
print("Matched FAQ:", matched_faq)
print("Answer:", matched_answer)
