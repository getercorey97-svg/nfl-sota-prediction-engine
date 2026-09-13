import os
import requests
import json

class AIFixer:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.target_files = []
        
        # Dynamically crawl the entire repository for relevant files
        for root, dirs, files in os.walk("."):
            # Exclude hidden directories (like .git, .github) and virtual envs to save context space
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
            
            for file in files:
                # Include Python, JSON, and YAML files
                if file.endswith(('.py', '.json', '.yml', '.yaml')):
                    if file == os.path.basename(__file__):
                        continue # Skip this exact script to prevent it from rewriting itself poorly
                    
                    filepath = os.path.relpath(os.path.join(root, file), ".")
                    self.target_files.append(filepath)

        # Force include the workflow files explicitly just in case
        workflow_dir = ".github/workflows"
        if os.path.exists(workflow_dir):
            for file in os.listdir(workflow_dir):
                if file.endswith(('.yml', '.yaml')):
                    self.target_files.append(f"{workflow_dir}/{file}")

        # Remove duplicates
        self.target_files = list(set(self.target_files))
            
        self.models = [
            "openrouter/free",                      
            "nvidia/nemotron-3-ultra-550b-a55b:free", 
            "poolside/laguna-s-2.1:free"             
        ]

    def run_system_audit(self):
        if not self.api_key:
            print("❌ Error: OPENROUTER_API_KEY is missing in GitHub Secrets.")
            return

        if not self.target_files:
            print("❌ Error: No valid files found to audit. Exiting process.")
            return

        print(f"📂 Loading {len(self.target_files)} files for universal audit: {', '.join(self.target_files)}")
        context = ""
        for file_name in self.target_files:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    context += f"\n--- START OF FILE: {file_name} ---\n"
                    context += f.read()
                    context += f"\n--- END OF FILE: {file_name} ---\n"
            except Exception as e:
                print(f"⚠️ Warning: Could not read {file_name} - {str(e)}")

        if not context.strip():
            print("❌ Error: Context is empty. Exiting process.")
            return

        # NEW UNIVERSAL PROMPT MAPPED TO THE 404 CRASH & 2026 COLUMNS
        prompt = f"""
        You are the Lead Architect and Autonomous AI Self-Healing Agent for a State-of-the-Art NFL Prediction Engine.
        Review this entire multi-file system for Python accuracy, JSON validity, YAML workflow syntax, runtime bugs, and data schema mismatches.
        
        SYSTEM CONTEXT:
        {context}
        
        REQUIRED FIXES (UNIVERSAL AUDIT):
        1. CRITICAL 404 CRASH FIX: The nfl_data_py package throws an HTTP 404 error when attempting to fetch data for the year 2026 because the library is deprecated. You MUST rewrite the `safe_load` functions and any data-fetching logic in `seed.py`, `backtest.py`, and `pipeline.py` to iterate through years INDIVIDUALLY inside a `try...except` block. If a year fails, print a warning and `continue`. Do NOT pass the whole list of years to the API at once.
        2. SCHEMA MIGRATION: Ensure all depth chart data dynamically looks for the new 2026 column names ('pos_abb', 'pos_name', 'pos_rank') alongside the old ones ('position', 'rank').
        3. Do NOT alter any existing predictive mathematics (Dixon-Coles, Copulas, Poisson logic, Backtesting logic, etc.) as they are already calibrated and correct. Only fix structural, data engineering, and code-level execution errors.
        4. Return the fully corrected code for ONLY the files that required changes. If a file is perfect, omit it from your response.
        
        OUTPUT INSTRUCTIONS:
        Return ONLY a JSON object where keys are the exact filenames provided in the context and values are the full, corrected raw code/data.
        Do not include markdown formatting like ```json.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/getercorey97-svg/nfl-sota-prediction-engine",
            "Content-Type": "application/json"
        }

        success = False
        for model in self.models:
            if success: break
            try:
                print(f"🤖 Requesting Universal Audit from: {model}...")
                payload = {
                    "model": model,
                    "messages": [{"role": "system", "content": "You are a self-healing coding agent that outputs pure, raw JSON. Do not include conversational text or markdown blocks."},
                                 {"role": "user", "content": prompt}]
                }
                
                res = requests.post(self.url, headers=headers, json=payload)
                
                if res.status_code != 200:
                    print(f"⚠️ Model {model} HTTP Error {res.status_code}: {res.text}")
                    continue
                    
                response = res.json()
                
                if 'error' in response:
                    print(f"⚠️ Model {model} failed: {response['error'].get('message')}")
                    continue

                if 'choices' not in response:
                    print(f"⚠️ Model {model} returned an invalid response structure.")
                    continue

                content = response['choices'][0]['message']['content'].strip()
                
                # Clean up markdown formatting if the AI ignores instructions
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                try:
                    fixes = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Model {model} did not return valid JSON. Error: {e}")
                    continue
                
                if not fixes:
                    print(f"✅ System check passed. {model} found no errors requiring changes.")
                    success = True
                    break

                for file_name, new_code in fixes.items():
                    # Clean paths just in case the AI prepends './'
                    clean_file_name = file_name.removeprefix('./') 
                    if clean_file_name in self.target_files:
                        with open(clean_file_name, "w", encoding="utf-8") as f:
                            f.write(new_code)
                        print(f"🛠️ AI successfully fixed: {clean_file_name}")
                    else:
                        print(f"⚠️ Warning: Model attempted to modify unrecognized file {file_name}.")
                
                success = True
                print(f"✅ System-wide universal audit complete using {model}.")
                    
            except Exception as e:
                print(f"⚠️ Attempt with {model} failed: {str(e)}")

if __name__ == "__main__":
    AIFixer().run_system_audit()
