import requests
import json
import time
import os
import argparse
from typing import List, Dict, Any, Tuple

class RecursiveLLM:
    def __init__(self, 
                 model_name: str = "deepseek-r1:14b", 
                 ollama_api_url: str = "http://localhost:11434/api/generate",
                 max_recursion_depth: int = 3,
                 save_thoughts: bool = True,
                 thoughts_dir: str = "thought_process"):
        self.model_name = model_name
        self.ollama_api_url = ollama_api_url
        self.max_recursion_depth = max_recursion_depth
        self.save_thoughts = save_thoughts
        self.thoughts_dir = thoughts_dir
        
        if self.save_thoughts:
            os.makedirs(self.thoughts_dir, exist_ok=True)
            
        # Tracking for thought process
        self.session_id = int(time.time())
        self.thought_log = []
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call the Ollama API with a given prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"
    
    def log_thought(self, stage: str, input_data: Any, output_data: Any) -> None:
        """Log the thought process"""
        thought = {
            "timestamp": time.time(),
            "stage": stage,
            "input": input_data,
            "output": output_data
        }
        self.thought_log.append(thought)
        
        if self.save_thoughts:
            with open(f"{self.thoughts_dir}/session_{self.session_id}.jsonl", "a") as f:
                f.write(json.dumps(thought) + "\n")
    
    def decompose_problem(self, query: str) -> List[str]:
        """Break down a problem into sub-problems"""
        decomposition_prompt = f"""
        I need to break down the following question into smaller, more specific sub-questions that will help solve the overall problem.
        
        Original question: {query}
        
        Think step by step and identify 2-5 key sub-questions that need to be answered to fully address the original question.
        
        Format your response as a JSON list of strings, with each string being a sub-question.
        Example format: ["sub-question 1", "sub-question 2", "sub-question 3"]
        
        Only return the JSON list, nothing else.
        """
        
        decomposition_result = self.call_llm(decomposition_prompt, temperature=0.7)
        
        try:
            # Try to parse the result as JSON
            sub_questions = json.loads(decomposition_result)
            if not isinstance(sub_questions, list):
                raise ValueError("Result is not a list")
        except Exception as e:
            # If parsing fails, try to extract content from brackets
            import re
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, decomposition_result, re.DOTALL)
            if match:
                try:
                    sub_questions_str = f"[{match.group(1)}]"
                    sub_questions = json.loads(sub_questions_str)
                except:
                    # If that also fails, use a simple fallback
                    sub_questions = [query]
            else:
                # Fallback to treating the original query as the only sub-question
                sub_questions = [query]
        
        self.log_thought("decompose_problem", query, sub_questions)
        return sub_questions
    
    def generate_sub_prompts(self, sub_questions: List[str], context: str) -> List[str]:
        """Generate a prompt for each sub-question"""
        prompts = []
        
        for question in sub_questions:
            prompt = f"""
            I need a detailed answer to the following specific question:
            
            Question: {question}
            
            Context: This is part of a larger problem: {context}
            
            Please provide a clear, concise, and detailed answer to this specific question.
            Focus only on answering this sub-question, not the entire problem.
            """
            prompts.append(prompt)
        
        self.log_thought("generate_sub_prompts", {"sub_questions": sub_questions, "context": context}, prompts)
        return prompts
    
    def solve_sub_problems(self, prompts: List[str]) -> List[str]:
        """Get answers for each sub-problem"""
        answers = []
        
        for prompt in prompts:
            answer = self.call_llm(prompt)
            answers.append(answer)
        
        self.log_thought("solve_sub_problems", prompts, answers)
        return answers
    
    def aggregate_solutions(self, original_query: str, sub_questions: List[str], answers: List[str]) -> str:
        """Combine all answers into a cohesive response"""
        # Create context for aggregation
        context_parts = []
        for i, (question, answer) in enumerate(zip(sub_questions, answers)):
            context_parts.append(f"Sub-question {i+1}: {question}\nAnswer {i+1}: {answer}")
        
        context = "\n\n".join(context_parts)
        
        aggregation_prompt = f"""
        I need to synthesize the following sub-answers into a comprehensive response to the original question.
        
        Original question: {original_query}
        
        Here are the sub-questions and their answers:
        
        {context}
        
        Please provide a complete, coherent answer to the original question based on these inputs.
        Your answer should be well-structured and comprehensive.
        """
        
        final_answer = self.call_llm(aggregation_prompt)
        self.log_thought("aggregate_solutions", 
                         {"original_query": original_query, "sub_qa_pairs": list(zip(sub_questions, answers))}, 
                         final_answer)
        
        return final_answer
    
    def process_query(self, query: str, recursion_depth: int = 0) -> str:
        """Process a query with recursive problem solving"""
        self.log_thought("process_query_start", {"query": query, "recursion_depth": recursion_depth}, None)
        
        # Base case - if we hit max recursion depth, just answer directly
        if recursion_depth >= self.max_recursion_depth:
            direct_answer = self.call_llm(f"Please answer this question directly: {query}")
            self.log_thought("direct_answer_due_to_max_depth", query, direct_answer)
            return direct_answer
        
        # Step 1: Decompose the problem
        sub_questions = self.decompose_problem(query)
        
        # If we got only one sub-question, and it's very similar to the original query,
        # just answer directly to avoid infinite recursion
        if len(sub_questions) == 1 and self.similarity(query, sub_questions[0]) > 0.8:
            direct_answer = self.call_llm(f"Please answer this question directly: {query}")
            self.log_thought("direct_answer_due_to_similar_subquestion", 
                           {"query": query, "sub_question": sub_questions[0]}, 
                           direct_answer)
            return direct_answer
        
        # Step 2: Generate prompts for sub-problems
        sub_prompts = self.generate_sub_prompts(sub_questions, query)
        
        # Step 3: For each sub-problem, decide whether to solve directly or recurse
        sub_answers = []
        for i, (sub_q, sub_prompt) in enumerate(zip(sub_questions, sub_prompts)):
            # Check complexity to decide whether to recurse
            complexity_score = self.assess_complexity(sub_q)
            
            if complexity_score > 0.7 and recursion_depth < self.max_recursion_depth - 1:
                # Recurse for complex sub-problems
                self.log_thought("recursive_call_decision", 
                               {"sub_question": sub_q, "complexity": complexity_score}, 
                               "Recursing")
                sub_answer = self.process_query(sub_q, recursion_depth + 1)
            else:
                # Solve directly for simpler sub-problems
                self.log_thought("direct_call_decision", 
                               {"sub_question": sub_q, "complexity": complexity_score}, 
                               "Solving directly")
                sub_answer = self.call_llm(sub_prompt)
            
            sub_answers.append(sub_answer)
        
        # Step 4: Aggregate all answers
        final_answer = self.aggregate_solutions(query, sub_questions, sub_answers)
        
        self.log_thought("process_query_end", 
                       {"query": query, "recursion_depth": recursion_depth}, 
                       final_answer)
        
        return final_answer
    
    def similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity check to avoid recursion loops"""
        # This is a very simple implementation - in production, you might want to use
        # something more sophisticated like cosine similarity with embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def assess_complexity(self, question: str) -> float:
        """Assess the complexity of a question to decide if it needs recursion"""
        # You could implement a more sophisticated approach here,
        # possibly using the LLM itself to judge complexity
        
        # Simple heuristic based on question length and presence of certain keywords
        complexity_factors = [
            len(question) > 100,  # Long questions
            "explain" in question.lower(),
            "how" in question.lower(),
            "why" in question.lower(),
            "compare" in question.lower(),
            "analyze" in question.lower(),
            question.count(",") > 2,  # Multiple parts
            question.count("?") > 1  # Multiple questions
        ]
        
        complexity = sum(1 for factor in complexity_factors if factor) / len(complexity_factors)
        return complexity
    
    def get_thought_process(self) -> List[Dict]:
        """Return the full thought process log"""
        return self.thought_log
    
    def save_thought_process(self, filename: str) -> None:
        """Save the thought process to a file"""
        with open(filename, 'w') as f:
            json.dump(self.thought_log, f, indent=2)



def main(args):
    # Initialize the recursive LLM system
    system = RecursiveLLM(
        model_name=args.m if args.m else "deepseek-r1:14b",
        max_recursion_depth=args.r if args.r else 3,
        save_thoughts=args.s
    )
    
    # Process a sample query
    query = args.p if args.p else "What are the environmental and economic impacts of renewable energy adoption globally?"
    
    print(f"Processing query: {query}")
    result = system.process_query(query)
    
    print("\n=== Final Result ===")
    print(result)
    
    # Save the thought process
    if args.s:
        system.save_thought_process("example_thought_process.json")
        print("\nThought process saved to example_thought_process.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-r max recursion depth (1~4), -p prompt, -m model name, -s save thought process")
    parser.add_argument("-r", type=int, choices=range(1, 5), help="max recursion depth (1~4)")
    parser.add_argument("-p", type=str, help="prompt")
    parser.add_argument("-m", type=str, help="model name")
    parser.add_argument("-s", action="store_true", help="Enable saving thought process")
    args = parser.parse_args()
    main(args)