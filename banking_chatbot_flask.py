# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import os
#
# # Configure Gemini API
# API_KEY = "AIzaSyBmHDQdrJXINfZBpZxkx3dCgLg2B15D3TY"  # Replace with your API key
# genai.configure(api_key=API_KEY)
#
# # Banking FAQ knowledge base
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
#
# def list_available_gemini_models():
#     """List all available Gemini models"""
#     try:
#         print("Checking available Gemini models...")
#         models = []
#         for model in genai.list_models():
#             if 'generateContent' in model.supported_generation_methods:
#                 models.append(model.name)
#                 print(f"✓ {model.name}")
#         return models
#     except Exception as e:
#         print(f"Error listing models: {e}")
#         return []
#
#
# class GeminiBankingChatbot:
#     def __init__(self):
#         print("=== Initializing Banking Chatbot with Gemini API ===")
#
#         # Load sentence transformer
#         print("Loading sentence transformer...")
#         try:
#             self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
#             print("✓ Sentence transformer loaded")
#         except Exception as e:
#             print(f"✗ Error loading sentence transformer: {e}")
#             return
#
#         # Initialize Gemini model
#         self.gemini_model = None
#         self.init_gemini_model()
#
#         # Prepare FAQ data
#         self.questions = list(banking_qa.keys())
#         self.answers = list(banking_qa.values())
#
#         # Pre-compute embeddings
#         print("Computing embeddings for FAQ database...")
#         try:
#             self.question_embeddings = self.encoder.encode(self.questions)
#             print("✓ Embeddings computed")
#         except Exception as e:
#             print(f"✗ Error computing embeddings: {e}")
#             return
#
#         print("✓ Banking Chatbot initialized successfully!\n")
#
#     def init_gemini_model(self):
#         """Initialize Gemini model with current 2025 model names"""
#         # Current Gemini model names as of 2025
#         model_names = [
#             'gemini-2.5-flash',  # Latest 2025 model
#             'gemini-2.5-pro',  # Latest 2025 pro model
#             'gemini-2.0-flash-exp',  # Experimental 2.0 model
#             'models/gemini-2.5-flash',  # With models/ prefix
#             'models/gemini-2.5-pro',
#             'models/gemini-2.0-flash-exp',
#             'gemini-pro',  # Fallback older model
#             'models/gemini-pro'
#         ]
#
#         for model_name in model_names:
#             try:
#                 print(f"Trying to initialize: {model_name}")
#                 test_model = genai.GenerativeModel(model_name)
#
#                 # Test with a simple prompt
#                 test_response = test_model.generate_content(
#                     "Say 'Hello' if you can respond.",
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=0.1,
#                         max_output_tokens=50
#                     )
#                 )
#
#                 if test_response and test_response.text:
#                     self.gemini_model = test_model
#                     self.model_name = model_name
#                     print(f"✓ Successfully initialized: {model_name}")
#                     return
#
#             except Exception as e:
#                 print(f"✗ Failed to initialize {model_name}: {str(e)}")
#                 continue
#
#         print("✗ Could not initialize any Gemini model")
#         print("Available models:")
#         list_available_gemini_models()
#
#     def find_relevant_faqs(self, user_query, top_k=3):
#         """Find most relevant FAQs using semantic similarity"""
#         try:
#             # Encode user query
#             query_embedding = self.encoder.encode([user_query])
#
#             # Calculate similarities
#             similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
#
#             # Get top-k most similar
#             top_indices = np.argsort(similarities)[-top_k:][::-1]
#
#             relevant_faqs = []
#             for idx in top_indices:
#                 if similarities[idx] > 0.3:  # Relevance threshold
#                     relevant_faqs.append({
#                         'question': self.questions[idx],
#                         'answer': self.answers[idx],
#                         'similarity': similarities[idx]
#                     })
#
#             return relevant_faqs
#
#         except Exception as e:
#             print(f"Error finding relevant FAQs: {e}")
#             return []
#
#     def generate_response(self, user_query):
#         """Generate response using Gemini API"""
#         try:
#             # Find relevant FAQs
#             relevant_faqs = self.find_relevant_faqs(user_query)
#
#             if not relevant_faqs:
#                 return "I don't have specific information about that query. Please contact our customer service team for personalized assistance."
#
#             if not self.gemini_model:
#                 # Fallback to direct answer
#                 return f"Based on our FAQ database:\n\n{relevant_faqs[0]['answer']}\n\nFor more assistance, please contact customer service."
#
#             # Build context for Gemini
#             context = ""
#             for i, faq in enumerate(relevant_faqs, 1):
#                 context += f"FAQ {i}:\nQ: {faq['question']}\nA: {faq['answer']}\n\n"
#
#             # Create prompt
#             prompt = f"""You are a helpful and professional banking customer service assistant.
#
# Based on the following relevant information from our FAQ database, please provide a clear, helpful response to the customer's question:
#
# {context}
#
# Customer Question: {user_query}
#
# Instructions:
# - Provide a conversational, helpful response
# - Use the FAQ information to answer accurately
# - Be concise but complete
# - Maintain a professional yet friendly tone
# - If the question isn't fully covered, suggest contacting customer service"""
#
#             # Generate response
#             response = self.gemini_model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.3,
#                     max_output_tokens=300,
#                     top_p=0.8
#                 )
#             )
#
#             if response and response.text:
#                 return response.text.strip()
#             else:
#                 return f"Based on our records:\n\n{relevant_faqs[0]['answer']}\n\nFor additional help, please contact our customer service."
#
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             # Fallback response
#             relevant_faqs = self.find_relevant_faqs(user_query)
#             if relevant_faqs:
#                 return f"I can help with that:\n\n{relevant_faqs[0]['answer']}\n\nIf you need further assistance, please contact customer service."
#             return "I apologize for the technical issue. Please contact our customer service for immediate assistance."
#
#
# def main():
#     print("=== Banking Customer Service Chatbot ===")
#     print("Powered by Google Gemini API")
#     print("Type 'quit', 'exit', or 'bye' to end the conversation.")
#     print("Type 'help' to see what I can assist with.\n")
#
#     # Initialize chatbot
#     chatbot = GeminiBankingChatbot()
#
#     if not chatbot.gemini_model:
#         print("Warning: Gemini model not initialized. Using fallback responses.")
#
#     # Chat loop
#     while True:
#         try:
#             user_input = input("You: ").strip()
#
#             if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
#                 print("\nThank you for using our banking chatbot. Have a great day!")
#                 break
#
#             if user_input.lower() == 'help':
#                 print("\nI can help you with:")
#                 print("• Password reset and account access")
#                 print("• Checking account balance")
#                 print("• Lost or stolen card issues")
#                 print("• Opening new accounts")
#                 print("• Loan applications and EMI schedules")
#                 print("• UPI and mobile banking setup")
#                 print("• Transaction disputes")
#                 print("• Account closure procedures")
#                 print("• Bank working hours and policies")
#                 continue
#
#             if not user_input:
#                 print("Please enter your question.")
#                 continue
#
#             print("\nBot:", chatbot.generate_response(user_input))
#
#         except KeyboardInterrupt:
#             print("\n\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"\nAn error occurred: {e}")
#             print("Please try again or contact customer service.")
#
#
# if __name__ == "__main__":
#     main()

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Configure Gemini API
API_KEY = "AIzaSyBmHDQdrJXINfZBpZxkx3dCgLg2B15D3TY"  # Replace with your API key
genai.configure(api_key=API_KEY)

# Banking FAQ knowledge base
banking_qa = {
    "How can I reset my online banking password?": "To reset your online banking password, visit the bank's website, click 'Forgot Password', enter your username or account number, and follow the instructions sent to your registered email or mobile.",
    "How do I check my account balance?": "You can check your account balance through online banking, mobile banking app, ATM, or by calling customer service.",
    "What should I do if my debit card is lost?": "If your debit card is lost, immediately call the customer service hotline to block the card. You can then request a new card through online banking or by visiting a branch.",
    "How do I activate international transactions on my credit card?": "To activate international transactions, log into online banking, go to Card Services, select your credit card, and enable international transactions. You may need to verify with OTP.",
    "How can I open a new savings account?": "To open a new savings account, visit any branch with required documents (ID proof, address proof, photographs) or apply online through the bank's website.",
    "What is the minimum balance required?": "The minimum balance requirement varies by account type. Regular savings accounts typically require ₹10,000, while zero-balance accounts have no minimum requirement.",
    "How do I update my registered mobile number?": "You can update your registered mobile number by visiting a branch with ID proof, calling customer service, or through online banking after OTP verification.",
    "How can I apply for a home loan?": "To apply for a home loan, visit a branch or apply online. You'll need income proof, property documents, and identity documents. The bank will assess your eligibility and process the application.",
    "What is the process for closing my bank account?": "To close your bank account, visit the branch with your account closure form, ID proof, and checkbook. Clear all dues and maintain minimum balance until closure.",
    "How do I check my loan EMI schedule?": "You can check your loan EMI schedule through online banking, mobile app, or by requesting a statement from the branch or customer service.",
    "How can I download my account statement?": "Download your account statement through online banking, mobile app, or request it from any branch. You can also get it emailed to your registered email address.",
    "What is the daily withdrawal limit from an ATM?": "The daily ATM withdrawal limit is typically ₹50,000, but this may vary based on your account type and card variant.",
    "How do I enable UPI payments?": "To enable UPI payments, download your bank's UPI app or any UPI-enabled app, register with your mobile number linked to your bank account, and create a UPI PIN.",
    "Can I increase my credit card limit?": "You can request a credit card limit increase through online banking, mobile app, customer service, or by visiting a branch. The bank will review your credit profile.",
    "What is the process to block a stolen credit card?": "To block a stolen credit card, immediately call the 24/7 customer service hotline. You can also block it through online banking or mobile app.",
    "How can I register for mobile banking?": "Register for mobile banking by downloading the bank's official app, using your account number and registered mobile number, and creating login credentials.",
    "How do I apply for a personal loan?": "To apply for a personal loan, visit a branch or apply online with income proof, identity documents, and employment details. The bank will assess your eligibility.",
    "What is the penalty for not maintaining minimum balance?": "The penalty for not maintaining minimum balance varies but typically ranges from ₹100-₹750 per month, depending on the shortfall amount.",
    "How can I dispute a wrong transaction?": "To dispute a wrong transaction, immediately contact customer service, file a complaint through online banking, or visit a branch with transaction details and proof.",
    "What are the bank's working hours?": "Bank branches are typically open Monday to Friday 10:00 AM to 4:00 PM, and Saturdays 10:00 AM to 2:00 PM. ATMs operate 24/7."
}


def list_available_gemini_models():
    # """List all available Gemini models"""
    try:
        print("Checking available Gemini models...")
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
                print(f"✓ {model.name}")
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


class GeminiBankingChatbot:
    def __init__(self):
        print("=== Initializing Banking Chatbot with Gemini API ===")

        # Load sentence transformer
        print("Loading sentence transformer...")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence transformer loaded")
        except Exception as e:
            print(f"✗ Error loading sentence transformer: {e}")
            return

        # Initialize Gemini model
        self.gemini_model = None
        self.init_gemini_model()

        # Prepare FAQ data
        self.questions = list(banking_qa.keys())
        self.answers = list(banking_qa.values())

        # Pre-compute embeddings
        print("Computing embeddings for FAQ database...")
        try:
            self.question_embeddings = self.encoder.encode(self.questions)
            print("✓ Embeddings computed")
        except Exception as e:
            print(f"✗ Error computing embeddings: {e}")
            return

        # Build FAISS index
        print("Building FAISS index for FAQ embeddings...")
        try:
            dimension = self.question_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product index

            # Normalize embeddings to use cosine similarity with inner product
            faiss.normalize_L2(self.question_embeddings)

            self.index.add(self.question_embeddings)
            print("✓ FAISS index built successfully")
        except Exception as e:
            print(f"✗ Error building FAISS index: {e}")
            return

        print("✓ Banking Chatbot initialized successfully!\n")

    def init_gemini_model(self):
        """Initialize Gemini model with current 2025 model names"""
        model_names = [
            'gemini-2.5-flash',  # Latest 2025 model
            'gemini-2.5-pro',  # Latest 2025 pro model
            'gemini-2.0-flash-exp',  # Experimental 2.0 model
            'models/gemini-2.5-flash',  # With models/ prefix
            'models/gemini-2.5-pro',
            'models/gemini-2.0-flash-exp',
            'gemini-pro',  # Fallback older model
            'models/gemini-pro'
        ]

        for model_name in model_names:
            try:
                print(f"Trying to initialize: {model_name}")
                test_model = genai.GenerativeModel(model_name)

                # Test with a simple prompt
                test_response = test_model.generate_content(
                    "Say 'Hello' if you can respond.",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=50
                    )
                )

                if test_response and test_response.text:
                    self.gemini_model = test_model
                    self.model_name = model_name
                    print(f"✓ Successfully initialized: {model_name}")
                    return

            except Exception as e:
                print(f"✗ Failed to initialize {model_name}: {str(e)}")
                continue

        print("✗ Could not initialize any Gemini model")
        print("Available models:")
        list_available_gemini_models()

    def find_relevant_faqs(self, user_query, top_k=3):
        """Find most relevant FAQs using FAISS similarity search"""
        try:
            # Encode user query
            query_embedding = self.encoder.encode([user_query])

            # Normalize query embedding
            faiss.normalize_L2(query_embedding)

            # Search top-k most similar in FAISS index
            distances, indices = self.index.search(query_embedding, top_k)

            relevant_faqs = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                similarity = distances[0][i]
                if similarity > 0.3:  # relevance threshold
                    relevant_faqs.append({
                        'question': self.questions[idx],
                        'answer': self.answers[idx],
                        'similarity': similarity
                    })
            return relevant_faqs

        except Exception as e:
            print(f"Error finding relevant FAQs with FAISS: {e}")
            return []

    def generate_response(self, user_query):
        """Generate response using Gemini API"""
        try:
            # Find relevant FAQs
            relevant_faqs = self.find_relevant_faqs(user_query)

            if not relevant_faqs:
                return "I don't have specific information about that query. Please contact our customer service team for personalized assistance."

            if not self.gemini_model:
                # Fallback to direct answer
                return f"Based on our FAQ database:\n\n{relevant_faqs[0]['answer']}\n\nFor more assistance, please contact customer service."

            # Build context for Gemini
            context = ""
            for i, faq in enumerate(relevant_faqs, 1):
                context += f"FAQ {i}:\nQ: {faq['question']}\nA: {faq['answer']}\n\n"

            # Create prompt
            prompt = f"""You are a helpful and professional banking customer service assistant. 

Based on the following relevant information from our FAQ database, please provide a clear, helpful response to the customer's question:

{context}

Customer Question: {user_query}

Instructions:
- Provide a conversational, helpful response
- Use the FAQ information to answer accurately
- Be concise but complete
- Maintain a professional yet friendly tone
- If the question isn't fully covered, suggest contacting customer service"""

            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                    top_p=0.8
                )
            )

            if response and response.text:
                return response.text.strip()
            else:
                return f"Based on our records:\n\n{relevant_faqs[0]['answer']}\n\nFor additional help, please contact our customer service."

        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            relevant_faqs = self.find_relevant_faqs(user_query)
            if relevant_faqs:
                return f"I can help with that:\n\n{relevant_faqs[0]['answer']}\n\nIf you need further assistance, please contact customer service."
            return "I apologize for the technical issue. Please contact our customer service for immediate assistance."


def main():
    print("=== Banking Customer Service Chatbot ===")
    print("Powered by Google Gemini API")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'help' to see what I can assist with.\n")

    # Initialize chatbot
    chatbot = GeminiBankingChatbot()

    if not chatbot.gemini_model:
        print("Warning: Gemini model is not initialized. Responses will be based on static FAQ answers only.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'bye'):
            print("Chatbot: Thank you for using the banking chatbot. Have a great day!")
            break

        if user_input.lower() == 'help':
            print("Chatbot: You can ask questions like:")
            print("- How can I reset my online banking password?")
            print("- What is the daily withdrawal limit from an ATM?")
            print("- How do I apply for a personal loan?")
            continue

        # Generate response
        answer = chatbot.generate_response(user_input)
        print(f"Chatbot: {answer}\n")


if __name__ == "__main__":
    main()



