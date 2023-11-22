from openai import OpenAI
import json

#If you are using .env
#from dotenv import load_dotenv
#load_dotenv() #load the .env file


#Extract everything for config_file having json
with open('config.json', "r") as config_file:
    config = json.load(config_file)

print(config)
client = OpenAI(api_key=config["OPENAIKEY"])


#BASIC 
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": 'what is 2+2', #what we will change
#         }
#     ],
#         model="gpt-4",
#     )
    
# print(chat_completion.choices[0].message.content) #extracting the content from hoices hold the variety of option



#storing the prompts of the client
history = []

while True: #will run forever
    new_prompt = input("Ask me something (use '$' to end conversation): ")

    if new_prompt == "$": #if user types $, then end the conversation
        print("Bye!")
        break

    # Format the history as a string
    history_string = str(history)

    # Append the history to your new prompt
    prompt_with_history = new_prompt + history_string

    # Create the completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
             "role": "user",
            "content": prompt_with_history
            }
        ],
        model="gpt-4",
    )

    response = chat_completion.choices[0].message.content
    history.append({
        new_prompt: response
    })
    print(f'Friend: {response}')



#STEP 2 : PUT IN ON A WEBSITE USING GRADIO OR APP USING STREAMLIT
#STEP 3 : FEED IT WITH LANGCHAIN OR EXTRA CSV MATERIALS
