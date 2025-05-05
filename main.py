import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re




# API KEY
load_dotenv()
client = OpenAI(api_key="------")

# load csv file into a dataframe
df = pd.read_csv("PROMISE_exp.csv")
# Ground Truth label ->  create TrueLabel (Ground Truth) it's job is just turn the multi-class labels into binary N/NF in order to do the binary classification task. 
df["TrueLabel"] = df["_class_"].apply(lambda x: "F" if x == "F" else "NF") 
 
# prompts
prompts = {
    "Cognitive Verifier": "Classify into functional (F) or non-functional (NF). Ask questions if needed.\n\nRequirement: ",
    "Context Manager": "Classify into functional (F) or non-functional (NF). Explain your reasoning.\n\nRequirement: ",
    "Persona": "Act as a requirements engineering expert. Classify into F or NF.\n\nRequirement: ",
    "Question Refinement": "Classify into functional (F) or non-functional (NF). Suggest better versions if needed.\n\nRequirement: ",
    "Template": "Read the requirement and return only F or NF.\n\nRequirement: "
}

# GPT helper function for software  artifact generation 
def ask_chatgpt(prompt, temperature=0.2, max_tokens=500):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# extracts F/NF from CHATGPT MODEL RESPONSE
def extract_label(text):
    if isinstance(text, str):
        match = re.search(r'\b(F|NF)\b', text)
        if match:
            return match.group(1)
    return "ERROR"


#SAMPLE SIZE OF REQUIREMENTS, where n is the # of requirements. currently it's set to 50
df_req = df.sample(n=50).reset_index(drop=True)

IS_GENERATE_ARTIFACTS_RUNNING = True # boolean to generate software artifacts based on software requirement 
IS_ARTIFACT_GENERATED = False #generate only 1 artifact to save tokens
for prompt_pattern, prompt_instruction in prompts.items():  
    for trial in range(1, 2):  # number of trials. currently it's 1
        df_req[prompt_pattern] = "" #place holder for the classification value for each prompt. 
        
        #df = df_all.sample(n=50).reset_index(drop=True)
        #artifact_generated = False  

        for i, row in df_req.iterrows():
            print(f"{prompt_pattern}: Requirement {i+1}/{len(df_req)}")  
            req_text = row["RequirementText"] # actual requirement 

            try:
                full_prompt = prompt_instruction + req_text
                messages = [
                    {"role": "system", "content": "You are a requirements engineering expert."},
                    {"role": "user", "content": full_prompt}
                ]
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=100
                )
                chatgpt_output = response.choices[0].message.content.strip() #chatgpt response 
                classification = extract_label(chatgpt_output) # chatgptcalling the extract label function and sending chatgpt's response to it. so it can extract F/NF from chatgpt's response and is assigned to classification variable

            except Exception as e:
                print("Error. Cannot be classified")
                classification = "ERROR"

            
            df_req.at[i, prompt_pattern] = classification # stores the classification of the samples chosen in the prompt pattern row/col

            #  software Artifact generation -> only generated once to save tokens
            if IS_GENERATE_ARTIFACTS_RUNNING and not IS_ARTIFACT_GENERATED: # currently set to TRUE
                if (classification == "F") and (row["TrueLabel"] == "F"):
                    try:
                        software_artifact_file = open(f"software_artifacts_{prompt_pattern.replace(' ', '_')}_trial_{trial}.txt", "w", encoding="utf-8")

                        use_case_prompt = f"""
                        You are a software analyst. Analyze the following requirement and reason step-by-step to understand its intent. Then, write a detailed use case.
                        Requirement: {req_text}
                        Step-by-step reasoning:
                        1. Explain the functionality or intent of the requirement.
                        2. Identify the user/actor.
                        3. Describe the system behavior.
                        4. Mention any conditions or exceptions or external dependencies even if not stated directly.
                        Now write the use case:

                        - Use case name:
                        - Primary actor:
                        - Goal:
                        - Preconditions:
                        - Main success scenario:
                        - Alternate flows:
                        - Postconditions:
                        """
                        use_case = ask_chatgpt(use_case_prompt) #calling the chatgpt helper function. the argument of this function is the use case prompt. returns chatgpt's response which is assigned to the use_case variable

                        class_diagram_prompt = f"You are a software designer. Based on the use case below, generate a UML-style class diagram.\n\nUse Case:\n{use_case}"
                        class_diagram = ask_chatgpt(class_diagram_prompt)

                        code_prompt = f"You are a software engineer. Based on this class diagram and use case, generate the Python code implementing the functionality.\n\nClass Diagram:\n{class_diagram}\n\nUse Case:\n{use_case}"
                        code = ask_chatgpt(code_prompt, max_tokens=700)

                        

                        software_artifact_file.write(f"\n--- PROMPT PATTERN: {prompt_pattern} ---\n")
                        software_artifact_file.write("\n--- REQUIREMENT ---\n" + req_text + "\n")
                        software_artifact_file.write("\n--- USE CASE ---\n" + use_case + "\n")
                        software_artifact_file.write("\n--- CLASS DIAGRAM ---\n" + class_diagram + "\n")
                        software_artifact_file.write("\n--- PYTHON CODE ---\n" + code + "\n")
                        software_artifact_file.write("\n" + "="*50 + "\n")

                        software_artifact_file.close()
                        IS_ARTIFACT_GENERATED = True
                        

                    except Exception as e:
                        print(f"Error generating software artifacts: {e}")

     
df_req.to_csv("binary_classification_total_results.csv", index=False)




        
      


