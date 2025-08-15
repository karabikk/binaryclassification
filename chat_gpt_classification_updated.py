import openai
import pandas as pd
import re
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix





def extract_label(text):
    lines = [line.strip().lower() for line in text.strip().splitlines() if line.strip()]
    last_lines = lines[-3:]  

    for line in reversed(last_lines):
        if "**nf**" in line or "non-functional" in line or line == "nf":
            return "NF"
        elif "**f**" in line or "functional" in line or line == "f":
            return "F"

    print(f"{last_lines}")
    return "Error"


api_key = ""
client = openai.OpenAI(api_key=api_key)




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
        "\nYou are a Requirements Engineer. Carefully analyze the requirement using step-by-step reasoning:\n"
        "1. Break down what the requirement is describing.\n"
        "2. Identify any expected system behavior.\n"
        "3. Identify any system qualities, constraints, or performance expectations.\n"
        "4. Decide whether the requirement is focused on what the system does or how it performs.\n"
        "5. Classify it as F or NF.\n\n"
        "Requirement: "
    ),
    "Question Refinement": (
        "\nCarefully analyze the following requirement and classify it as Functional (F) or Non-Functional (NF) using step-by-step reasoning:\n"
        "1. Briefly explain what the requirement is describing."
        "2. Determine whether it focuses on system behavior or qualities/constraints."
        "3. Rewrite the requirement only if it is vague, ambiguous, or underspecified — otherwise, proceed with the original wording.\n"
        "4. Conclude with your final classification of F or NF:\n"
        "Requirement: "
    )
}



results = []

for index, row in sampled_requirements.iterrows():
    requirement_text = row["RequirementText"]
    ground_truth = row["GroundTruth"]
    for prompt_name, prompt_template in prompts.items():
        full_prompt = prompt_template + requirement_text
        print(prompt_name + ": " + full_prompt)
        print()
        try:
           
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a requirements engineering expert."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
              
            )

            model_output = response.choices[0].message.content.strip()
            print(model_output)
            classification = extract_label(model_output) 
            print(f"Extracted Label: {classification}")
            print("---------------------------------------------------------------------------------------------------------------------")
            results.append({
                "requirement_id": index + 1,
                "requirement": requirement_text,
                "prompt_type": prompt_name,
                "classification": classification,
                "ground_truth": ground_truth   

                
            })
            

          
        
        except Exception as error:
            print(f"{error}")
            continue
        
        


        
  

results_df = pd.DataFrame(results)
results_df.to_csv("classification_total_results.csv", index=False)


for prompt_name in prompts:
    y_true = results_df[results_df["prompt_type"] == prompt_name]["ground_truth"]
    y_pred = results_df[results_df["prompt_type"] == prompt_name]["classification"]

    precision = precision_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="F", average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["F", "NF"])

    print(f"{prompt_name}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print("Confusion Matrix:\n", pd.DataFrame(cm, index=["Actual F", "Actual NF"], columns=["Pred F", "Pred NF"]))


