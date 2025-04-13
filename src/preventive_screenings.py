from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

print(f"Loaded API Key: {os.environ.get('GOOGLE_API_KEY')}")

# Create a ChatOpenAI model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25",temperature=0.0)


goal = """You are a physician expert at the US preventive task force group which publishes screening guidelines for preventive health.

Your goal is to output a list of screening tests that the patient should have performed based on the provided clinical information.

Make these recommendations based on a combination of the patient's age, gender, weight, their past medical history, their family history, their social history and lifestyle,
their geographic location, their ethnicity, their diet, their exercise habits, their smoking history, their sexual history, their alcohol use, their tobacco use, their drug use,
and their travel history.  You may consider other relevant factors too, such as their blood work, their lab tests, their imaging studies, and current medications taken, 
if available and applicable. 

If a patient has already had a screening test performed, you should not recommend it again if the test is not yet due based on the clinical guidelines. On the other hand,
if it has been too long since their last screening, point that out and recommend that it is time to get screened again.

It is crucial that you only make recommendations based on the provided clinical guidelines. You also must tell the doctor which evidence supports your 
recommendations with a link to the information or give your reasoning in one sentence that is no more than 25 words. 

Use URL cloaking for any links to the source of the guidelines. The link to the source of the guidelines should just be labeled 'source'.

Rank the recommendations based on the provided clinical information.  The more the recommendation is based on clinical information (other than age and gender), the higher it should be ranked.

Give recommendations using the following format:
1. <screening test name>
        <reason>
2. <screening test name>
        <reason>
3. <screening test name>
        <reason>

Here is an example response:
1. Colonoscopy
        Reason: All adults 45 years old and older should be screened for colorectal cancer once every 10 years.  She has never been screened before.
             
2. Mammogram
        Reason: All females 45 years old and older should have a screening mammogram every two years.  She was last screened 4 years ago (normal result)


If you can not make any recommendations based on the clinical guidelines and the patient's clinical information, state that you can not make any recommendations."""

# Define age and gender
patient_age = 50
patient_gender = "female"

# Template for patient specific info
# Use single braces for .format() substitution
patient_info_template = """The patient's age is {age} years old.  The patient's gender is: {gender}. The patient smokes, drinks 5 beers per day.  
She had a low-dose CT scan of her chest six months ago that was normal. Her last colonoscopy was 1 year ago, and it was normal.
She reports feeling sad and without purpose after her husband passed away six months ago.  She is not eating well and is sleeping very little.
The patient's mother died of breast cancer at age 45. The patient has a black eye, and bruises on her arms."""

# Format the patient info string using the defined age and gender
formatted_patient_info = patient_info_template.format(age=patient_age, gender=patient_gender)

# Define the message structure using the templates
messages = [("system", goal),
           ("human", "Which screening tests does my patient need based on the clinical guidelines? Here is their information: {patient_info}"),
]

# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Prompt for Preventive Screening (Tuple) -----\n")
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     ("human", "Tell me {joke_count} jokes."),
# ]
prompt_template = ChatPromptTemplate.from_messages(messages)

# Invoke the template, providing values for all placeholders
# The 'goal' template doesn't use 'age' or 'gender'
# The 'human' template uses 'patient_info' (which we provide pre-formatted)
prompt = prompt_template.invoke({
    # "age": patient_age, # Not needed as 'goal' template doesn't use {age}
    # "gender": patient_gender, # Not needed as 'goal' template doesn't use {gender}
    "patient_info": formatted_patient_info # Pass the formatted string here
})
print(prompt)
result = model.invoke(prompt)
print(result.content)
