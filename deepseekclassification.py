import openai
import pandas as pd
import re
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import json





def extract_label(text):
 
    NF_PATTERN = re.compile(r'\*{0,2}\b(?:NF|non[-\s]?functional)\b\*{0,2}', re.IGNORECASE)
    F_PATTERN  = re.compile(r'\*{0,2}\b(?:F|functional)\b\*{0,2}', re.IGNORECASE)

    lines = text.splitlines()
    start_index = None

  
    for i, line in enumerate(lines):
        if re.match(r'^\s*5\.', line):
            start_index = i
            break

 
    if start_index is None:
        for i, line in enumerate(lines):
            if re.match(r'^\s*4\.', line):
                start_index = i
                break

    
    if start_index is not None:
        content_lines = [l for l in lines[start_index:] if l.strip()]
        first_three_text = "\n".join(content_lines[:5])

        if NF_PATTERN.search(first_three_text):
            return "NF"
        elif F_PATTERN.search(first_three_text):
            return "F"
        else:
            return "F"  
    else:
        return "F"  




client = openai.OpenAI(api_key="", base_url="https://api.deepseek.com")




df = pd.read_csv("PROMISE_exp.csv")



df["GroundTruth"] = df["_class_"].apply(lambda label: "F" if label == "F" else "NF")

#sampled_requirements = df.sample(n=10).reset_index(drop=True)
sampled_requirements = df.reset_index(drop=True)

# Prompts
prompts = {
    "Cognitive Verifier": (
        "\nCarefully examine the following requirement to determine whether it is Functional (F) or Non-Functional (NF).\n"
        "Step by step, do the following:\n"
        "1. Birefly summarize what the requirement is describing.\n"
        "2. Determine whether it describes what the system should do or how it should be.\n"
        "3. If it contains both aspects, identify which one is emphasized.\n"
        "4. Based on your reasoning, classify it as F or NF.\n"
        "Requirement: "
    ),
    "Persona": (
        "\nYou are a Requirements Engineer.\n"
        "1. Break down what the requirement is describing.\n"
        "2. Identify any expected system behavior.\n"
        "3. Identify any system qualities, constraints, or performance expectations.\n"
        "4. Decide whether the requirement is focused on what the system does or how it performs.\n"
        "5. Conclude with your final classification of F or NF.\n\n"
        "Requirement: "
    ),
    "Question Refinement": (
        "\n Carefully analyze the following requirement and classify it as Functional (F) or Non-Functional (NF) using step-by-step reasoning:\n"
        "First, Briefly explain what the requirement is describing. Then Determine whether it focuses on system behavior or qualities/constraints. Then check for clarity and rewrite the requirement only if it is vague, ambiguous, or underspecified — otherwise, proceed with the original wording.\n"
        "Finally conclude with your final classification of F or NF:\n"
        "Requirement: "
    )
}



results = []

resultsWithOutput = []


for index, row in sampled_requirements.iterrows():
    requirement_text = row["RequirementText"]
    ground_truth = row["GroundTruth"]
    for prompt_name, prompt_template in prompts.items():
        full_prompt = prompt_template + requirement_text
        print(prompt_name + ": " + full_prompt)
        print()
        try:
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a requirements engineering expert."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
              
            )

            model_output = response.choices[0].message.content.strip()
            print(model_output)
            classification = extract_label(model_output) 
            print(f" {classification}")
            print("---------------------------------------------------------------------------------------------------------------------")
            results.append({
                "requirement_id": index + 1,
                "requirement": requirement_text,
                "prompt_type": prompt_name,
                "classification": classification,
                "ground_truth": ground_truth   

                
            })
            resultsWithOutput.append(
                {
                     "requirement": requirement_text,
                     "prompt_type": prompt_name,
                     "ground_truth": ground_truth,
                     "model_response": model_output,
                     "ground_truth": ground_truth 

                }
            )
            

          
        
        except Exception as error:
            print(f" {error}")
            continue
        
        


        
  


results_df = pd.DataFrame(results)
results_df.to_csv("deekseek_classification_total_results.csv", index=False)

results_model_response = pd.DataFrame(resultsWithOutput)
results_model_response.to_csv("modelresoinse.csv", index=False)



with open("modelresoinse.json", "w", encoding="utf-8") as f:
    json.dump(resultsWithOutput, f, indent=2, ensure_ascii=False)


for prompt_name in prompts:
    y_true = results_df[results_df["prompt_type"] == prompt_name]["ground_truth"]
    y_pred = results_df[results_df["prompt_type"] == prompt_name]["classification"]

    precision = precision_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["F", "NF"])

    print(f"\n {prompt_name}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print("Confusion Matrix:\n", pd.DataFrame(cm, index=["Actual F", "Actual NF"], columns=["Pred F", "Pred NF"]))

