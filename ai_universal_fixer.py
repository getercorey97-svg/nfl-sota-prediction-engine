import os
import glob
import requests
import json

class AIFixer:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Dynamically aggregate all Python files in the root directory, excluding this script
        self.target_files = [f for f in glob.glob("*.py") if f != os.path.basename(__file__)]
        
        # Explicitly include the metadata file
        if os.path.exists("engine_metadata.json"):
            self.target_files.append("engine_metadata.json")
            
        # Explicitly include the workflow file
        workflow_file = ".github/workflows/nfl_meta_engine.yml"
        if os.path.exists(workflow_file):
            self.target_files.append(workflow_file)
            
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

        print(f"📂 Loading {len(self.target_files)} files for audit: {', '.join(self.target_files)}")
        context = ""
        for file_name in self.target_files:
            try:
                with open(file_name, "r") as f:
                    context += f"\n--- START OF FILE: {file_name} ---\n"
                    context += f.read()
                    context += f"\n--- END OF FILE: {file_name} ---\n"
            except Exception as e:
                print(f"⚠️ Warning: Could not read {file_name} - {str(e)}")

        if not context.strip():
            print("❌ Error: Context is empty. Exiting process.")
            return

        # Prompt updated: Removed completed tasks to prevent unwanted regressions
        prompt = f"""
        You are the Lead Architect for a State-of-the-Art NFL Prediction Engine.
        Review this entire multi-file system for Python accuracy, JSON validity, and YAML workflow syntax.
        
        SYSTEM CONTEXT:
        {context}
        
        REQUIRED FIXES:
        1. CRITICAL FIX: In pipeline.py, resolve the "KeyError: 'position'" crash on the starters DataFrame. 
        2. Standardize the columns in pipeline.py to lowercase before filtering, and add a fallback check to verify if 'position', 'pos', or similar keys exist in starters.columns to prevent fatal crashes if upstream data formatting changes.
        3. Do NOT alter any existing predictive mathematics (Dixon-Coles, Copulas, etc.) as they are already calibrated and correct.
        
        OUTPUT INSTRUCTIONS:
        Return ONLY a JSON object where keys are the exact filenames provided in the context and values are the full corrected code/data.
        Do not include markdown formatting like ```json.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "[https://github.com/getercorey97-svg/nfl-sota-prediction-engine](https://github.com/getercorey97-svg/nfl-sota-prediction-engine)",
            "Content-Type": "application/json"
        }

        success = False
        for model in self.models:
            if success: break
            try:
                print(f"🤖 Requesting Audit from: {model}...")
                payload = {
                    "model": model,
                    "messages": [{"role": "system", "content": "You are a self-healing coding agent that outputs raw JSON."},
                                 {"role": "user", "content": prompt}]
                }
                
                res = requests.post(self.url, headers=headers, json=payload)
                response = res.json()
                
                if 'error' in response:
                    print(f"⚠️ Model {model} failed: {response['error'].get('message')}")
                    continue

                if 'choices' not in response:
                    print(f"⚠️ Model {model} returned an invalid response structure.")
                    continue

                content = response['choices'][0]['message']['content'].strip()
                
                # Clean up markdown formatting
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                fixes = json.loads(content)
                
                for file_name, new_code in fixes.items():
                    if file_name in self.target_files:
                        with open(file_name, "w") as f:
                            f.write(new_code)
                        print(f"🛠️ AI successfully fixed {file_name}")
                    else:
                        print(f"⚠️ Warning: Model attempted to modify unauthorized file {file_name}.")
                
                success = True
                print(f"✅ System-wide audit complete using {model}.")
                    
            except Exception as e:
                print(f"⚠️ Attempt with {model} failed: {str(e)}")

if __name__ == "__main__":
    AIFixer().run_system_audit()
