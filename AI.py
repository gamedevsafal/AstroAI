import asyncio
import os
import uuid
import gradio as gr
import sys

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content

# Astrologer agent: asks for either birthdate or zodiac sign, gives Barnum statements
reply_maker_agent = LlmAgent(
    name="astrologer_bot",
    model="gemini-2.5-flash-lite",
    instruction="""
You are an astrologer chatbot. If the user provides their birthdate (in dd-mm-yyyy format), zodiac sign (as a word, e.g., Aries, Taurus, Gemini), or a question, give them advice related to question they asked.
If the user does not provide any of these or with a question of not your type, still reply with a general, friendly advice related to the question in best ways possible in easiest words.
Never get stuck asking for input—always respond with something encouraging, simple, and easy to understand and different senteces everytime.
Never break character as an astrologer.
If the user asks about payment, kindly remind them to use the payment button below for a detailed reading.
"""
)

runner = None
session_id = None

async def get_agent_response(user_text: str, history=None) -> str:
    """Gets a response from the AstrologerBot agent, with history context if no history context just get response."""
    global runner, session_id

    if runner is None:
        runner = InMemoryRunner(
            agent=reply_maker_agent,
            app_name="astrologer-app"
        )
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user123",
            session_id=session_id,
        )

    # Build context from full history for memory
    context = ""
    if history:
        for user, bot in history:
            context += f"User: {user}\nAstrologer: {bot}\n"
    context += f"User: {user_text}\nAstrologer:"

    new_message = Content(parts=[{"text": context}])
    events = runner.run(
        user_id="user123",
        session_id=session_id,
        new_message=new_message
    )

    reply_text = ""
    for event in events:
        if event.is_final_response():
            reply_text = event.content.parts[0].text
            break
    return reply_text

def launch_gradio_interface(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key

    def chat_fn(message, history):
        return asyncio.run(get_agent_response(message, history))

    with gr.Blocks(title="Astrologer Bot", theme=gr.themes.Soft()) as demo:
        # Custom CSS for appealing look
        gr.HTML("""
        <style>
        body { background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%); }
        .gradio-container { border-radius: 16px; box-shadow: 0 4px 24px #0001; }
        #main-title { color: #6d28d9; font-size: 2.2em; font-weight: bold; }
        #subtitle { color: #312e81; font-size: 1.2em; margin-bottom: 1em; }
        #payment-btn { background: #f59e42; color: white; font-weight: bold; border-radius: 8px; }
        #publish-btn { background: #10b981; color: white; font-weight: bold; border-radius: 8px; }
        </style>
        """)

        # Title and subtitle
        gr.Markdown('<div id="main-title">🌟 Astrologer Bot</div>')
        gr.Markdown('<div id="subtitle">Get your personalized reading! Please enter your birthdate (dd-mm-yyyy) or zodiac sign (e.g., Aries, Taurus, Gemini) and your question.<br>For a detailed reading, use the payment button below.</div>')

        # Login/Signup form (placeholder)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Textbox(label="Username", elem_id="username-box")
                gr.Textbox(label="Password", type="password", elem_id="password-box")
                gr.Button("Login / Signup", elem_id="login-btn")
        
        # Payment gateway placeholder (Stripe/Razorpay link)
        gr.Markdown("#### Payment")
        gr.Button("Pay for Detailed Reading", elem_id="payment-btn")
        gr.Markdown('[Pay securely with Stripe](https://buy.stripe.com/test_00g5nQ4vI5gQfWc3cc) _(Demo link)_')

        # Publish button (placeholder)
        gr.Button("Publish Website", elem_id="publish-btn")
        gr.Markdown("_Publishing integration coming soon. This is a placeholder button._")

        # ChatGPT-like interface
        gr.ChatInterface(
            chat_fn,
            chatbot=gr.Chatbot(show_label=False, bubble_full_width=False, avatar="🪐"),
            textbox=gr.Textbox(placeholder="Type your birthdate, zodiac sign, and question...")
        )

    print("Launching Astrologer Bot interface...")
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AI.py <YOUR_GOOGLE_API_KEY>")
        sys.exit(1)
    api_key_from_args = sys.argv[1]
    asyncio.run(launch_gradio_interface(api_key_from_args))