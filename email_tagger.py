import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from typing import List, Dict
from collections import defaultdict
import re


class EmailTagger:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the email tagger with customer isolation
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"🔧 Loading model: {model_name}")
        self.prompt_cache = {}
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📍 Using device: {self.device}")
        
        # Load model with 4-bit quantization to save VRAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("✅ Model loaded successfully!")
        
        # Customer-specific tag mappings
        self.customer_tags = defaultdict(set)
        
    def build_customer_tag_catalog(self, df: pd.DataFrame):
        """Build customer-specific tag catalog for isolation"""
        for _, row in df.iterrows():
            self.customer_tags[row['customer_id']].add(row['tag'])
        
        print("\n📊 Customer Tag Catalog:")
        for customer, tags in self.customer_tags.items():
            print(f"  {customer}: {len(tags)} tags -> {sorted(tags)}")
    
    def create_classification_prompt(self, subject: str, body: str, 
                                    customer_id: str, 
                                    few_shot_examples: List[Dict] = None) -> str:
        """
        Create a few-shot prompt for email classification with customer isolation
        """
        # Get valid tags for this customer only
        valid_tags = sorted(self.customer_tags[customer_id])
        
        prompt = f"""You are an email classification system for customer support tickets.

**IMPORTANT**: You can ONLY use tags that belong to customer {customer_id}.

Valid tags for {customer_id}: {', '.join(valid_tags)}

Classify the following email into ONE of the valid tags above.

"""
        
        # Add few-shot examples if provided
        if few_shot_examples:
            prompt += "Here are some example classifications:\n\n"
            for ex in few_shot_examples:
                prompt += f"Subject: {ex['subject']}\n"
                prompt += f"Body: {ex['body']}\n"
                prompt += f"Tag: {ex['tag']}\n\n"
        
        prompt += f"""Now classify this email:

Subject: {subject}
Body: {body}

Respond with ONLY the tag name, nothing else."""
        
        return prompt
    
    def predict_tag(self, subject: str, body: str, customer_id: str, 
                   few_shot_examples: List[Dict] = None) -> str:
        """Predict tag for an email"""
        
        # Input validation
        if not subject and not body:
            return "unknown"
        
        if customer_id not in self.customer_tags:
            raise ValueError(f"Unknown customer_id: {customer_id}")
        
        # Check cache
        key = (subject, body, customer_id)
        if key in self.prompt_cache:
            prompt = self.prompt_cache[key]
        else:
            prompt = self.create_classification_prompt(
                subject, body, customer_id, few_shot_examples
            )
            self.prompt_cache[key] = prompt
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract tag from response (get everything after the prompt)
        # Remove the prompt part to get only the model's response
        prompt_text = prompt.replace("Respond with ONLY the tag name, nothing else.", "").strip()
        predicted_text = response[len(prompt_text):].strip()
        
        # Clean and extract the tag
        # Remove common prefixes like "Tag:", "Answer:", etc.
        predicted_text = re.sub(r'^(tag|answer|response|classification):\s*', '', predicted_text, flags=re.IGNORECASE)
        
        # Extract only alphanumeric and underscore characters
        predicted_tag = re.sub(r'[^a-z_]', '', predicted_text.lower())
        
        # Validate against customer's allowed tags
        valid_tags = self.customer_tags[customer_id]
        if predicted_tag not in valid_tags:
            # Try to find closest match
            for valid_tag in valid_tags:
                if valid_tag in predicted_text.lower() or predicted_tag in valid_tag:
                    predicted_tag = valid_tag
                    break
            else:
                # If still no match, return the first valid tag as fallback
                predicted_tag = list(valid_tags)[0] if valid_tags else "unknown"
        
        return predicted_tag
    
    def apply_patterns_and_guardrails(self, subject: str, body: str, 
                                     predicted_tag: str) -> Dict:
        """
        Apply pattern matching and anti-pattern guardrails
        
        Patterns: Keyword-based confidence boosting
        Anti-patterns: Common misclassification detection
        """
        text = f"{subject} {body}".lower()
        
        # Define patterns (words that strongly indicate certain tags)
        patterns = {
            'billing': ['charge', 'invoice', 'payment', 'bill', 'cost'],
            'access_issue': ['permission', 'denied', 'access', 'login', 'authenticate'],
            'performance': ['slow', 'delay', 'lag', 'freeze', 'seconds'],
            'feature_request': ['feature', 'request', 'would like', 'want', 'need'],
            'automation_bug': ['automation', 'duplicate', 'workflow', 'creating'],
            'status_bug': ['stuck', 'pending', 'resolved', 'status'],
            'workflow_issue': ['rule', 'workflow', 'stopped working'],
            'tagging_issue': ['tag', 'tagging', 'not appearing'],
            'analytics_issue': ['csat', 'analytics', 'dashboard', 'disappeared'],
            'setup_help': ['setup', 'configure', 'guide', 'help setting'],
            'mail_merge_issue': ['mail merge', 'failing', 'not sending'],
            'user_management': ['add user', 'team member', 'authorization'],
        }
        
        # Define anti-patterns (misleading words)
        anti_patterns = {
            'billing_in_product': ['product.*bill', 'bill.*feature'],
            'access_in_automation': ['automation.*access', 'workflow.*permission']
        }
        
        # Check patterns
        pattern_match = any(kw in text for kw in patterns.get(predicted_tag, []))
        
        # Check anti-patterns
        anti_pattern_detected = any(
            re.search(regex, text)
            for regex in anti_patterns.get(predicted_tag, [])
        )
        
        confidence = 0.5
        if pattern_match:
            confidence += 0.3
        if anti_pattern_detected:
            confidence -= 0.4
        
        confidence = max(0.1, min(confidence, 1.0))
        
        return {
            'predicted_tag': predicted_tag,
            'pattern_match': pattern_match,
            'anti_pattern_detected': anti_pattern_detected,
            'confidence': round(confidence, 2)
        }


def main():
    """Main execution function for Part A"""
    
    # Load data
    print("📂 Loading email data...")
    df = pd.read_csv('data/emails_small.csv')
    
    # Initialize tagger
    tagger = EmailTagger()
    
    # Build customer tag catalog
    tagger.build_customer_tag_catalog(df)
    
    # Split data: use first 2 emails per customer as few-shot examples
    customer_examples = defaultdict(list)
    test_data = []
    
    for customer in df['customer_id'].unique():
        customer_df = df[df['customer_id'] == customer]
        examples = customer_df.head(2).to_dict('records')
        customer_examples[customer] = examples
        test_data.extend(customer_df.tail(len(customer_df) - 2).to_dict('records'))
    
    print(f"\n🧪 Testing on {len(test_data)} emails...")
    
    # Run predictions
    results = []
    correct = 0
    total = 0
    
    for email in test_data:
        few_shot = customer_examples[email['customer_id']]
        
        predicted = tagger.predict_tag(
            email['subject'], 
            email['body'], 
            email['customer_id'],
            few_shot
        )
        
        # Apply guardrails
        result = tagger.apply_patterns_and_guardrails(
            email['subject'], 
            email['body'], 
            predicted
        )
        
        ground_truth = email['tag']
        is_correct = predicted == ground_truth
        
        results.append({
            'email_id': email['email_id'],
            'customer_id': email['customer_id'],
            'subject': email['subject'],
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct,
            **result
        })
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f"[Email {email['email_id']}] GT={ground_truth} | Pred={predicted} | Conf={result['confidence']} | {'✓' if is_correct else '✗'}")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Save results
    with open('outputs/part_a_results.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': results
        }, f, indent=2)
    
    print("\n✅ Part A complete! Results saved to outputs/part_a_results.json")


if __name__ == "__main__":
    main()
