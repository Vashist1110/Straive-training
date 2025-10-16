from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure Gemini API key (replace with your key)
API_KEY = "AIzaSyBmHDQdrJXINfZBpZxkx3dCgLg2B15D3TY"
genai.configure(api_key=API_KEY)

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

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

class GeminiFinanceChatbot:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.questions = list(finance_qa.keys())
        self.answers = list(finance_qa.values())
        self.question_embeddings = self.encoder.encode(self.questions, convert_to_numpy=True)

        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.question_embeddings)
        self.index.add(self.question_embeddings)

        try:
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            print("Gemini model init failed:", e)
            self.gemini_model = None

    def find_relevant_faqs(self, user_query, top_k=3):
        query_embedding = self.encoder.encode([user_query], convert_to_numpy=True)
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
