import openai
from dotenv import load_dotenv
import os
import json

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key

# Define file paths
file_path = r'C:\Users\gagss\OneDrive\Documents\spring 2024\Deep learning on Cloud Platforms\Project\test.jsonl'
output_simple_file_path = r'C:\Users\gagss\OneDrive\Documents\spring 2024\Deep learning on Cloud Platforms\Project\output_simple.json'
output_correct_context_file_path = r'C:\Users\gagss\OneDrive\Documents\spring 2024\Deep learning on Cloud Platforms\Project\output_correct_context.json'
output_wrong_date_context_file_path = r'C:\Users\gagss\OneDrive\Documents\spring 2024\Deep learning on Cloud Platforms\Project\output_wrong_date_context.json'
output_irrelevant_context_file_path = r'C:\Users\gagss\OneDrive\Documents\spring 2024\Deep learning on Cloud Platforms\Project\output_irrelevant_context.json'

# Initialize output lists
output_simple_list = []
output_correct_context_list = []
output_wrong_date_context_list = []
output_irrelevant_context_list = []

# Read data from the JSONL file
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line as a JSON object
        row = json.loads(line)

        # Extract information from the JSON object
        i = row.get('i')
        question = row.get('question')
        relevant_context = row.get('relevant_context')
        wrong_date_context = row.get('wrong_date_context')
        irrelevant_relevant_context = row.get('random_context')

        try:
            # Define the common system message to act as a helpful assistant
            system_message = {
                "role": "system",
                "content": "Act as a helpful assistant and answer these questions to the best of your ability."
            }

            # 1. Simply asking the model
            prompt_simple = f"Q: {question}\nA:"
            response_simple = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    system_message,
                    {"role": "user", "content": prompt_simple}
                ],
                temperature=0.7,
                max_tokens=100
            )
            output_simple = response_simple.choices[0].message.content.strip()
            output_simple_list.append({"index": i, "output": output_simple})

            # 2. Asking the model with correct context
            prompt_correct_context = f"Q: {question}\nC: {relevant_context}\nA:"
            response_correct_context = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    system_message,
                    {"role": "user", "content": prompt_correct_context}
                ],
                temperature=0.7,
                max_tokens=100
            )
            output_correct_context = response_correct_context.choices[0].message.content.strip()
            output_correct_context_list.append({"index": i, "output": output_correct_context})

            # 3. Asking model with incorrect date context
            prompt_wrong_date_context = f"Q: {question}\nC: {wrong_date_context}\nA:"
            response_wrong_date_context = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    system_message,
                    {"role": "user", "content": prompt_wrong_date_context}
                ],
                temperature=0.7,
                max_tokens=100
            )
            output_wrong_date_context = response_wrong_date_context.choices[0].message.content.strip()
            output_wrong_date_context_list.append({"index": i, "output": output_wrong_date_context})

            # 4. Asking the model with irrelevant context
            prompt_irrelevant_context = f"Q: {question}\nC: {irrelevant_relevant_context}\nA:"
            response_irrelevant_context = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    system_message,
                    {"role": "user", "content": prompt_irrelevant_context}
                ],
                temperature=0.7,
                max_tokens=100
            )
            output_irrelevant_context = response_irrelevant_context.choices[0].message.content.strip()
            output_irrelevant_context_list.append({"index": i, "output": output_irrelevant_context})

            # Print the outputs directly
            print(f"Index: {i}")
            print(f"Output Simple: {output_simple}")
            print(f"Output Correct Context: {output_correct_context}")
            print(f"Output Wrong Date Context: {output_wrong_date_context}")
            print(f"Output Irrelevant Context: {output_irrelevant_context}")
            print()  # Blank line for readability between questions

        except Exception as e:
            print(f"Error processing question {i}: {e}")

# Save each type of output to separate JSON files
with open(output_simple_file_path, 'w') as json_file:
    json.dump(output_simple_list, json_file, indent=4)

with open(output_correct_context_file_path, 'w') as json_file:
    json.dump(output_correct_context_list, json_file, indent=4)

with open(output_wrong_date_context_file_path, 'w') as json_file:
    json.dump(output_wrong_date_context_list, json_file, indent=4)

with open(output_irrelevant_context_file_path, 'w') as json_file:
    json.dump(output_irrelevant_context_list, json_file, indent=4)

print("All outputs saved to separate files.")
