# from flask import Flask, request, jsonify
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import faiss
#
# # Configure Gemini API
# API_KEY = "AIzaSyBmHDQdrJXINfZBpZxkx3dCgLg2B15D3TY"
# genai.configure(api_key=API_KEY)
#
# # Banking FAQ
# banking_qa = {
#
#     "How can I reset my online banking password?": "To reset your online banking password, visit the bank's website, click 'Forgot Password', enter your username or account number, and follow the instructions sent to your registered email or mobile.",
#     "How do I check my account balance?": "You can check your account balance through online banking, mobile banking app, ATM, or by calling customer service.",
#     "What should I do if my debit card is lost?": "If your debit card is lost, immediately call the customer service hotline to block the card. You can then request a new card through online banking or by visiting a branch.",
#     "How do I activate international transactions on my credit card?": "To activate international transactions, log into online banking, go to Card Services, select your credit card, and enable international transactions. You may need to verify with OTP.",
#     "How can I open a new savings account?": "To open a new savings account, visit any branch with required documents (ID proof, address proof, photographs) or apply online through the bank's website.",
#     "What is the minimum balance required?": "The minimum balance requirement varies by account type. Regular savings accounts typically require ₹10,000, while zero-balance accounts have no minimum requirement.",
#     "How do I update my registered mobile number?": "You can update your registered mobile number by visiting a branch with ID proof, calling customer service, or through online banking after OTP verification.",
#     "How can I apply for a home loan?": "To apply for a home loan, visit a branch or apply online. You'll need income proof, property documents, and identity documents. The bank will assess your eligibility and process the application.",
#     "What is the process for closing my bank account?": "To close your bank account, visit the branch with your account closure form, ID proof, and checkbook. Clear all dues and maintain minimum balance until closure.",
#     "How do I check my loan EMI schedule?": "You can check your loan EMI schedule through online banking, mobile app, or by requesting a statement from the branch or customer service.",
#     "How can I download my account statement?": "Download your account statement through online banking, mobile app, or request it from any branch. You can also get it emailed to your registered email address.",
#     "What is the daily withdrawal limit from an ATM?": "The daily ATM withdrawal limit is typically ₹50,000, but this may vary based on your account type and card variant.",
#     "How do I enable UPI payments?": "To enable UPI payments, download your bank's UPI app or any UPI-enabled app, register with your mobile number linked to your bank account, and create a UPI PIN.",
#     "Can I increase my credit card limit?": "You can request a credit card limit increase through online banking, mobile app, customer service, or by visiting a branch. The bank will review your credit profile.",
#     "What is the process to block a stolen credit card?": "To block a stolen credit card, immediately call the 24/7 customer service hotline. You can also block it through online banking or mobile app.",
#     "How can I register for mobile banking?": "Register for mobile banking by downloading the bank's official app, using your account number and registered mobile number, and creating login credentials.",
#     "How do I apply for a personal loan?": "To apply for a personal loan, visit a branch or apply online with income proof, identity documents, and employment details. The bank will assess your eligibility.",
#     "What is the penalty for not maintaining minimum balance?": "The penalty for not maintaining minimum balance varies but typically ranges from ₹100-₹750 per month, depending on the shortfall amount.",
#     "How can I dispute a wrong transaction?": "To dispute a wrong transaction, immediately contact customer service, file a complaint through online banking, or visit a branch with transaction details and proof.",
#     "What are the bank's working hours?": "Bank branches are typically open Monday to Friday 10:00 AM to 4:00 PM, and Saturdays 10:00 AM to 2:00 PM. ATMs operate 24/7."
# }
#
# # Initialize Flask app
# app = Flask(__name__)
#
#
# # Initialize the chatbot once on server start
# class GeminiBankingChatbot:
#     def __init__(self):
#         self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         self.questions = list(banking_qa.keys())
#         self.answers = list(banking_qa.values())
#         self.question_embeddings = self.encoder.encode(self.questions)
#
#         dimension = self.question_embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(dimension)
#         faiss.normalize_L2(self.question_embeddings)
#         self.index.add(self.question_embeddings)
#
#         # Initialize Gemini model
#         try:
#             self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # Or your preferred model
#         except Exception as e:
#             print("Gemini model init failed:", e)
#             self.gemini_model = None
#
#     def find_relevant_faqs(self, user_query, top_k=3):
#         query_embedding = self.encoder.encode([user_query])
#         faiss.normalize_L2(query_embedding)
#         distances, indices = self.index.search(query_embedding, top_k)
#         relevant = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx == -1 or dist < 0.3:
#                 continue
#             relevant.append({'question': self.questions[idx], 'answer': self.answers[idx], 'similarity': dist})
#         return relevant
#
#     def generate_response(self, user_query):
#         relevant_faqs = self.find_relevant_faqs(user_query)
#         if not relevant_faqs:
#             return "Sorry, I don't have information on that. Please contact customer support."
#
#         if not self.gemini_model:
#             return relevant_faqs[0]['answer']
#
#         context = ""
#         for i, faq in enumerate(relevant_faqs, 1):
#             context += f"FAQ {i}:\nQ: {faq['question']}\nA: {faq['answer']}\n\n"
#
#         prompt = f"""You are a helpful banking assistant.
#
# Based on the following FAQ information, answer the customer's question:
#
# {context}
#
# Customer Question: {user_query}
#
# Please provide a clear, professional, and concise response."""
#
#         try:
#             response = self.gemini_model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.3,
#                     max_output_tokens=300,
#                     top_p=0.8
#                 )
#             )
#             return response.text.strip() if response and response.text else relevant_faqs[0]['answer']
#         except Exception as e:
#             print("Error generating Gemini response:", e)
#             return relevant_faqs[0]['answer']
#
#
# chatbot = GeminiBankingChatbot()
#
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json(force=True)
#     user_query = data.get('query', '').strip()
#     if not user_query:
#         return jsonify({'error': 'No query provided'}), 400
#
#     answer = chatbot.generate_response(user_query)
#     return jsonify({'response': answer})
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
#
# # from flask import Flask, request, jsonify
# # from sentence_transformers import SentenceTransformer
# # import faiss
# # import numpy as np
# #
# # # ----------------------
# # # Step 1: Create Banking FAQs
# # # ----------------------
# # faqs = [
# #     "How can I reset my online banking password?",
# #     "How do I check my account balance?",
# #     "What should I do if my debit card is lost?",
# #     "How can I apply for a personal loan?",
# #     "How do I activate international transactions on my credit card?"
# # ]
# #
# # answers = {
# #     faqs[0]: "You can reset your password by clicking 'Forgot Password' on the login page and following the verification steps.",
# #     faqs[1]: "You can check your balance using the mobile app, internet banking, or by visiting an ATM.",
# #     faqs[2]: "Report the lost debit card immediately through the customer service helpline or the banking app.",
# #     faqs[3]: "You can apply for a personal loan online through the bank’s portal or by visiting your nearest branch.",
# #     faqs[4]: "International transactions can be activated via the mobile banking app or by contacting customer care."
# # }
# #
# # # ----------------------
# # # Step 2: Load Embedding Model
# # # ----------------------
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # faq_embeddings = model.encode(faqs)
# #
# # # ----------------------
# # # Step 3: Create FAISS Index
# # # ----------------------
# # dimension = faq_embeddings.shape[1]
# # index = faiss.IndexFlatL2(dimension)
# # index.add(np.array(faq_embeddings))
# #
# # # ----------------------
# # # Step 4: Initialize Flask
# # # ----------------------
# # app = Flask(__name__)
# #
# # @app.route('/')
# # def home():
# #     return "Welcome to the Banking FAQ Semantic Search API!"
# #
# # @app.route('/ask', methods=['POST'])
# # def ask_question():
# #     try:
# #         data = request.json
# #         user_query = data.get('question', '')
# #         if not user_query:
# #             return jsonify({"error": "Please provide a question in JSON format: {'question': 'your question here'}"}), 400
# #
# #         # Encode user query
# #         query_embedding = model.encode([user_query])
# #
# #         # Search FAISS index
# #         distances, indices = index.search(np.array(query_embedding), 1)
# #         first_match_index = indices[0][0]
# #         matched_faq = faqs[first_match_index]
# #         matched_answer = answers[matched_faq]
# #
# #         return jsonify({
# #             "user_question": user_query,
# #             "matched_faq": matched_faq,
# #             "answer": matched_answer,
# #             "distance": float(distances[0][0])
# #         })
# #
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500
# #
# # if __name__ == '__main__':
# #     app.run(debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
#
# # Banking FAQ data
# banking_qa = {
#     "How can I reset my online banking password?": "To reset your online banking password, visit the bank's website, click 'Forgot Password', enter your username or account number, and follow the instructions sent to your registered email or mobile.",
#     "How do I check my account balance?": "You can check your account balance through online banking, mobile banking app, ATM, or by calling customer service.",
#     "What should I do if my debit card is lost?": "If your debit card is lost, immediately call the customer service hotline to block the card. You can then request a new card through online banking or by visiting a branch.",
#     "How do I activate international transactions on my credit card?": "To activate international transactions, log into online banking, go to Card Services, select your credit card, and enable international transactions. You may need to verify with OTP.",
#     "How can I open a new savings account?": "To open a new savings account, visit any branch with required documents (ID proof, address proof, photographs) or apply online through the bank's website.",
#     "What is the minimum balance required?": "The minimum balance requirement varies by account type. Regular savings accounts typically require ₹10,000, while zero-balance accounts have no minimum requirement.",
#     "How do I update my registered mobile number?": "You can update your registered mobile number by visiting a branch with ID proof, calling customer service, or through online banking after OTP verification.",
#     "How can I apply for a home loan?": "To apply for a home loan, visit a branch or apply online. You'll need income proof, property documents, and identity documents. The bank will assess your eligibility and process the application.",
#     "What is the process for closing my bank account?": "To close your bank account, visit the branch with your account closure form, ID proof, and checkbook. Clear all dues and maintain minimum balance until closure.",
#     "How do I check my loan EMI schedule?": "You can check your loan EMI schedule through online banking, mobile app, or by requesting a statement from the branch or customer service.",
#     "How can I download my account statement?": "Download your account statement through online banking, mobile app, or request it from any branch. You can also get it emailed to your registered email address.",
#     "What is the daily withdrawal limit from an ATM?": "The daily ATM withdrawal limit is typically ₹50,000, but this may vary based on your account type and card variant.",
#     "How do I enable UPI payments?": "To enable UPI payments, download your bank's UPI app or any UPI-enabled app, register with your mobile number linked to your bank account, and create a UPI PIN.",
#     "Can I increase my credit card limit?": "You can request a credit card limit increase through online banking, mobile app, customer service, or by visiting a branch. The bank will review your credit profile.",
#     "What is the process to block a stolen credit card?": "To block a stolen credit card, immediately call the 24/7 customer service hotline. You can also block it through online banking or mobile app.",
#     "How can I register for mobile banking?": "Register for mobile banking by downloading the bank's official app, using your account number and registered mobile number, and creating login credentials.",
#     "How do I apply for a personal loan?": "To apply for a personal loan, visit a branch or apply online with income proof, identity documents, and employment details. The bank will assess your eligibility.",
#     "What is the penalty for not maintaining minimum balance?": "The penalty for not maintaining minimum balance varies but typically ranges from ₹100-₹750 per month, depending on the shortfall amount.",
#     "How can I dispute a wrong transaction?": "To dispute a wrong transaction, immediately contact customer service, file a complaint through online banking, or visit a branch with transaction details and proof.",
#     "What are the bank's working hours?": "Bank branches are typically open Monday to Friday 10:00 AM to 4:00 PM, and Saturdays 10:00 AM to 2:00 PM. ATMs operate 24/7."
# }
#
# # Initialize Flask app and enable CORS
# app = Flask(__name__)
# CORS(app)
#
# # Initialize SentenceTransformer and FAISS index
# encoder = SentenceTransformer('all-MiniLM-L6-v2')
# questions = list(banking_qa.keys())
# answers = list(banking_qa.values())
#
# question_embeddings = encoder.encode(questions, convert_to_numpy=True)
# faiss.normalize_L2(question_embeddings)
#
# dimension = question_embeddings.shape[1]
# index = faiss.IndexFlatIP(dimension)  # Inner Product similarity (cosine after normalization)
# index.add(question_embeddings)
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json(force=True)
#     user_query = data.get('query', '').strip()
#     if not user_query:
#         return jsonify({'error': 'No query provided'}), 400
#
#     # Encode user query and search
#     query_embedding = encoder.encode([user_query], convert_to_numpy=True)
#     faiss.normalize_L2(query_embedding)
#
#     k = 1  # top 1 match
#     distances, indices = index.search(query_embedding, k)
#
#     best_idx = indices[0][0]
#     best_score = distances[0][0]
#
#     # Threshold for similarity
#     if best_score < 0.3:
#         return jsonify({'response': "Sorry, I don't have information on that. Please contact customer support."})
#
#     answer = answers[best_idx]
#     return jsonify({'response': answer})
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure Gemini API
API_KEY = "AIzaSyBmHDQdrJXINfZBpZxkx3dCgLg2B15D3TY"
genai.configure(api_key=API_KEY)

# Finance Management FAQ
finance_qa = {
    "How do I create a budget?": "Start by tracking your income and expenses, categorize your spending, set spending limits for each category, and review your budget monthly.",
    "What is an emergency fund?": "An emergency fund is money set aside to cover unexpected expenses, ideally covering 3-6 months of living costs.",
    "How can I improve my credit score?": "Pay bills on time, reduce outstanding debt, avoid opening too many new accounts, and keep old credit accounts open.",
    "What is the difference between saving and investing?": "Saving is setting money aside safely for short-term goals, while investing involves risking money for potential long-term growth.",
    "How do I start investing in stocks?": "Open a brokerage account, research stocks, start with diversified investments like ETFs, and invest consistently over time.",
    "What are mutual funds?": "Mutual funds pool money from many investors to buy a diversified portfolio of stocks, bonds, or other securities managed by professionals.",
    "How do I plan for retirement?": "Start saving early, contribute to retirement accounts like 401(k) or IRAs, diversify investments, and regularly review your retirement goals.",
    "What is compound interest?": "Compound interest is the interest earned on the initial principal plus the accumulated interest from previous periods.",
    "How can I reduce my debt?": "Create a repayment plan prioritizing high-interest debt, consider debt consolidation, and avoid accumulating new debt.",
    "What is a credit report?": "A credit report is a detailed summary of your credit history used by lenders to evaluate your creditworthiness.",
    "How do I track my expenses?": "Use budgeting apps, spreadsheets, or manual logs to record and categorize your daily spending.",
    "What is diversification in investing?": "Diversification means spreading investments across different assets to reduce risk.",
    "How do I set financial goals?": "Define specific, measurable, achievable, relevant, and time-bound (SMART) goals to guide your financial planning.",
    "What are the tax benefits of retirement accounts?": "Many retirement accounts offer tax deferrals or deductions on contributions and tax-free growth.",
    "How do I protect myself from identity theft?": "Monitor your credit reports, use strong passwords, avoid sharing sensitive information, and be cautious with emails and links.",
    "What is an IRA?": "An IRA (Individual Retirement Account) is a tax-advantaged retirement savings account available in several types like Traditional and Roth IRAs.",
    "How much should I save each month?": "Aim to save at least 20% of your income, adjusting based on your goals and expenses.",
    "What is a budget surplus?": "A budget surplus occurs when your income exceeds your expenses during a specific period.",
    "How can I automate my savings?": "Set up automatic transfers from your checking account to a savings or investment account on a regular schedule.",
    "What is the 50/30/20 rule?": "A budgeting rule that allocates 50% of income to needs, 30% to wants, and 20% to savings or debt repayment."
}

# Initialize Flask app
app = Flask(__name__)


class GeminiFinanceChatbot:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.questions = list(finance_qa.keys())
        self.answers = list(finance_qa.values())
        self.question_embeddings = self.encoder.encode(self.questions)

        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.question_embeddings)
        self.index.add(self.question_embeddings)

        try:
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # Or your preferred model
        except Exception as e:
            print("Gemini model init failed:", e)
            self.gemini_model = None

    def find_relevant_faqs(self, user_query, top_k=3):
        query_embedding = self.encoder.encode([user_query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist < 0.3:
                continue
            relevant.append({'question': self.questions[idx], 'answer': self.answers[idx], 'similarity': dist})
        return relevant

    def generate_response(self, user_query):
        relevant_faqs = self.find_relevant_faqs(user_query)
        if not relevant_faqs:
            return "Sorry, I don't have information on that. Please consult a financial advisor."

        if not self.gemini_model:
            return relevant_faqs[0]['answer']

        context = ""
        for i, faq in enumerate(relevant_faqs, 1):
            context += f"FAQ {i}:\nQ: {faq['question']}\nA: {faq['answer']}\n\n"

        prompt = f"""You are a helpful financial assistant.

Based on the following FAQ information, answer the customer's question:

{context}

Customer Question: {user_query}

Please provide a clear, professional, and concise response."""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                    top_p=0.8
                )
            )
            return response.text.strip() if response and response.text else relevant_faqs[0]['answer']
        except Exception as e:
            print("Error generating Gemini response:", e)
            return relevant_faqs[0]['answer']


chatbot = GeminiFinanceChatbot()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_query = data.get('query', '').strip()
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    answer = chatbot.generate_response(user_query)
    return jsonify({'response': answer})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

