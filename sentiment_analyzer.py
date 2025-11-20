import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import re
from typing import Dict

class SentimentAnalyzer:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        print(f"Loading sentiment analysis model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("Model loaded successfully!")
        
        self.valid_sentiments = ["positive", "negative", "neutral"]
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize email text"""
        if not text:
            return ""
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:1500]
    
    def create_sentiment_prompt_v1(self, subject: str, body: str) -> str:
        """Version 1: Simpler prompt with explicit output format"""
        return f"""Classify the sentiment of this customer support email.

Email Subject: {subject}
Email Body: {body}

Classify as one of: positive, negative, neutral

Output format:
sentiment: <positive|negative|neutral>
confidence: <0.0 to 1.0>
reasoning: <brief explanation>

Now classify:"""

    def create_sentiment_prompt_v2(self, subject: str, body: str) -> str:
        """Version 2: Enhanced prompt with definitions and examples"""
        return f"""You are a customer support sentiment classifier.

Sentiment definitions:
- positive: customer is satisfied, thankful, or appreciative
- negative: customer is frustrated, angry, or reporting problems
- neutral: customer is asking questions or providing information

Examples:
Email: "Thank you so much for helping!"
Output:
sentiment: positive
confidence: 0.9
reasoning: Expresses gratitude

Email: "Nothing works, system is broken!"
Output:
sentiment: negative
confidence: 0.95
reasoning: Clear frustration with issues

Email: "How do I setup automation?"
Output:
sentiment: neutral
confidence: 0.8
reasoning: Informational question

Email Subject: {subject}
Email Body: {body}

Classify using this format:
sentiment: <positive|negative|neutral>
confidence: <0.0 to 1.0>
reasoning: <brief explanation>

Now classify:"""

    def extract_sentiment_from_text(self, response: str) -> Dict:
        """Extract sentiment information from plain text response"""
        
        result = {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'reasoning': 'Unable to parse'
        }
        
        try:
            sentiment_match = re.search(r'sentiment:\s*(positive|negative|neutral)', response, re.IGNORECASE)
            if sentiment_match:
                result['sentiment'] = sentiment_match.group(1).lower()
            
            confidence_match = re.search(r'confidence:\s*(0\.\d+|1\.0|[0-9])', response, re.IGNORECASE)
            if confidence_match:
                conf_str = confidence_match.group(1)
                result['confidence'] = max(0.0, min(1.0, float(conf_str)))
            
            reasoning_match = re.search(r'reasoning:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            
        except Exception as e:
            print(f"  Warning: Parse error - {str(e)}")
        
        return result

    def analyze_sentiment(self, subject: str, body: str, prompt_version: int = 1) -> Dict:
        """Analyze sentiment using specified prompt version"""
        
        subject = self.preprocess(subject)
        body = self.preprocess(body)
        
        if prompt_version == 1:
            prompt = self.create_sentiment_prompt_v1(subject, body)
            temp = 0.3
            sampling = True
        else:
            prompt = self.create_sentiment_prompt_v2(subject, body)
            temp = 0.5
            sampling = True
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temp,
                    do_sample=sampling,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = self.extract_sentiment_from_text(response)
            result['raw_response'] = response[-300:]
            
        except Exception as e:
            result = {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'reasoning': f'Error during analysis',
                'raw_response': str(e)[:100]
            }
        
        return result

def main():
    """Main execution for Part B"""
    
    print("Loading sentiment test emails...")
    df = pd.read_csv('data/sentiment_emails.csv')
    
    analyzer = SentimentAnalyzer()
    
    results_v1 = []
    results_v2 = []
    
    print("\nTesting Prompt V1 (simple format)...")
    for idx, email in df.iterrows():
        result = analyzer.analyze_sentiment(email['subject'], email['body'], prompt_version=1)
        results_v1.append({
            'email_id': email['email_id'],
            'subject': email['subject'],
            'body': email['body'],
            **result
        })
        print(f"  Email {email['email_id']}: {result.get('sentiment')} (conf: {result.get('confidence'):.2f})")
    
    print("\nTesting Prompt V2 (enhanced with examples)...")
    for idx, email in df.iterrows():
        result = analyzer.analyze_sentiment(email['subject'], email['body'], prompt_version=2)
        results_v2.append({
            'email_id': email['email_id'],
            'subject': email['subject'],
            'body': email['body'],
            **result
        })
        print(f"  Email {email['email_id']}: {result.get('sentiment')} (conf: {result.get('confidence'):.2f})")
    
    sentiment_counts_v1 = pd.Series([r['sentiment'] for r in results_v1]).value_counts().to_dict()
    sentiment_counts_v2 = pd.Series([r['sentiment'] for r in results_v2]).value_counts().to_dict()
    
    avg_conf_v1 = sum(r.get('confidence', 0) for r in results_v1) / len(results_v1) if results_v1 else 0
    avg_conf_v2 = sum(r.get('confidence', 0) for r in results_v2) / len(results_v2) if results_v2 else 0
    
    output = {
        'prompt_v1_results': results_v1,
        'prompt_v2_results': results_v2,
        'analysis': {
            'v1_avg_confidence': round(avg_conf_v1, 3),
            'v2_avg_confidence': round(avg_conf_v2, 3),
            'v1_sentiment_distribution': sentiment_counts_v1,
            'v2_sentiment_distribution': sentiment_counts_v2,
            'total_emails_analyzed': len(df),
            'comparison': {
                'confidence_improvement': round(avg_conf_v2 - avg_conf_v1, 3),
                'v1_deterministic': 'Uses temperature and sampling',
                'v2_with_examples': 'Uses in-context examples'
            }
        }
    }
    
    with open('outputs/part_b_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"V1 Average Confidence: {avg_conf_v1:.2%}")
    print(f"V1 Sentiment Distribution: {sentiment_counts_v1}")
    print(f"\nV2 Average Confidence: {avg_conf_v2:.2%}")
    print(f"V2 Sentiment Distribution: {sentiment_counts_v2}")
    print(f"\nConfidence Improvement: {(avg_conf_v2 - avg_conf_v1):.2%}")
    print("="*60)
    
    print("\nPart B complete! Results saved to outputs/part_b_results.json")

if __name__ == "__main__":
    main()
