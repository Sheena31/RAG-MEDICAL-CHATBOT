from flask import Flask, render_template,request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from .env file
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup
def  nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route('/', methods=['GET', 'POST'])
def index():
    if "messages" not in session:
        session["messages"] = []
    
    if request.method == 'POST':  #Collecting prompt from user and store it in the user input variable
        user_input = request.form.get("prompt")

        if user_input:
            message = session["messages"] # Use the previous info from the session
            message.append({"role": "user", "content": user_input}) # User's message
            session["messages"] = message # Update the session with the new message

            try:
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({"query": user_input }) #generatin response with respect to the query
                result = response.get("result", "No answer found.") # Extract the result from the response

                message.append({"role": "assistant", "content": result})    #assistant's response
                session["messages"] = message # Update the session with the assistant's response
            except Exception as e:  
                error_msg = f"An error occurred: {str(e)}"  
                return render_template('index.html', messages = session["messages"], error = error_msg) #flask automatically detects the template folder
        
        return redirect(url_for('index')) # Redirect to the index route to display the updated messages
    return render_template('index.html', messages = session.get("messages", [])) 

@app.route('/clear')
def clear():
    session.pop("messages", None) # Clear the messages from the session
    return redirect(url_for('index')) # Redirect to the index route

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)