import torch
import os
import re
import json
import datetime
import numpy as np
from collections import defaultdict
import sys
from functools import lru_cache
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl', 'utils', 'reward_score'))
from math_dapo import (
    last_boxed_only_string,
    remove_boxed,
    normalize_final_answer,
    is_correct_minerva,
    is_correct_strict_box,
    verify,
    compute_score
)

class BatchProcessor:
    """Batch parallel processor - processes batch generation of same layer for multiple questions"""
    
    def __init__(self, model, tokenizer, llm, branch_count, max_branch_depth, rollout_times, heuristic_branch_interval, global_file_manager, batch_size=100, strict_box_verify=True):
        self.model = model
        self.tokenizer = tokenizer
        self.llm = llm
        self.branch_count = branch_count
        self.max_branch_depth = max_branch_depth
        self.rollout_times = rollout_times
        self.heuristic_branch_interval = heuristic_branch_interval
        self.global_file_manager = global_file_manager
        self.batch_size = batch_size
        self.strict_box_verify = strict_box_verify
        
        print(f"Batch Processor initialized with batch_size={batch_size}, branch_count={branch_count}, max_depth={max_branch_depth}")
    
    def process_batch(self, questions_batch, result_dir, model_name):
        """Process a batch of questions"""
        print(f"\nProcessing batch of {len(questions_batch)} questions...")
        
        question_states = {}
        for i, question_data in enumerate(questions_batch):
            question_id = question_data['question_id']
            question_text = question_data['question']
            solution = question_data['solution']
            prompt = question_data['prompt']
            
            try:
                current_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"Error creating initial text for question {question_id}: {e}")
                current_text = str(prompt)
            
            question_states[question_id] = {
                'question_text': question_text,
                'solution': solution,
                'current_text': current_text,
                'initial_length': len(current_text),
                'current_layer': [(current_text, "")],
                'branch_results': {},
                'log_content': [],
                'save_path': os.path.join(result_dir, f"{model_name}_{question_id}.txt")
            }
        
        # Iteratively process each layer
        for layer in range(self.max_branch_depth):
            print(f"\nLayer {layer}: Processing {len(questions_batch)} questions")
            
            # Collect all nodes in current layer for all questions
            all_expanded_responses = []
            all_question_ids = []
            all_branch_paths = []
            
            for question_id, state in question_states.items():
                current_layer = state['current_layer']
                
                if not current_layer:
                    continue  # This question has no active nodes
                
                # Expand current layer nodes for each question
                for response, path in current_layer:
                    for i in range(self.branch_count):
                        all_expanded_responses.append(response)
                        all_question_ids.append(question_id)
                        if path == "":
                            new_path = f"{i + 1}"
                        else:
                            new_path = f"{path}.{i + 1}"
                        all_branch_paths.append(new_path)
            
            if not all_expanded_responses:
                print(f"No active nodes in layer {layer}")
                break
            
            print(f"Batch generating {len(all_expanded_responses)} responses across {len(questions_batch)} questions...")
            
            next_layer_responses = self.batch_generate_next_layer(all_expanded_responses)
            
            # Process results grouped by question
            for question_id, state in question_states.items():
                current_layer = state['current_layer']
                if not current_layer:
                    continue
                
                question_start_idx = None
                question_end_idx = None
                
                for i, qid in enumerate(all_question_ids):
                    if qid == question_id:
                        if question_start_idx is None:
                            question_start_idx = i
                        question_end_idx = i + 1
                
                if question_start_idx is not None:
                    question_responses = all_expanded_responses[question_start_idx:question_end_idx]
                    question_paths = all_branch_paths[question_start_idx:question_end_idx]
                    question_new_responses = next_layer_responses[question_start_idx:question_end_idx]
                    
                    # Process results for this question
                    terminated_branches = []
                    non_terminated_branches = []
                    
                    for i, (original_response, path, new_text) in enumerate(zip(question_responses, question_paths, question_new_responses)):
                        full_text = original_response + new_text
                        should_terminate, termination_reason = self.check_branch_termination(full_text, state['initial_length'])
                        
                        if should_terminate:
                            is_correct, extracted_answer, status = self.verify_answer_correctness(full_text, path, state['solution'])
                            boxed_answer = self.extract_boxed_answer(full_text, state['initial_length'])
                            
                            state['branch_results'][path] = {
                                'boxed_answer': boxed_answer,
                                'extracted_answer': extracted_answer,
                                'is_correct': is_correct,
                                'termination_reason': termination_reason,
                                'status': status
                            }
                            
                            final_content = full_text[state['initial_length']:].strip()
                            completion_info = f" [BRANCH {path} COMPLETE - {termination_reason.upper()}] {status}"
                            state['log_content'].append(f"{final_content}{completion_info}")
                            
                            terminated_branches.append((full_text, path))
                        else:
                            non_terminated_branches.append((full_text, path))
                    
                    # Update next layer for this question
                    question_states[question_id]['current_layer'] = non_terminated_branches
                
                # If reached max depth, add to rollout queue
                if layer == self.max_branch_depth - 1 and non_terminated_branches:
                    for text, path in non_terminated_branches:
                        self.add_to_global_rollout_queue(text, path, question_id, state['solution'])
        
        # Process rollout queue for all questions
        self.process_global_rollout_queue(question_states)
        
        # Save results for each question
        for question_id, state in question_states.items():
            self.save_question_results(question_id, state)
        
        return question_states
    
    def batch_generate_next_layer(self, partial_responses):
        """Batch generate answers for next layer"""
        try:
            from vllm import SamplingParams
            
            print(f"Batch generating {len(partial_responses)} responses with {self.heuristic_branch_interval} tokens each...")
            
            sampling_params = SamplingParams(
                n=1,
                temperature=0.7,  # Fixed temperature
                top_p=1.0,        # Fixed top_p
                max_tokens=self.heuristic_branch_interval
            )
            
            outputs = self.llm.generate(partial_responses, sampling_params)
            
            all_responses = []
            for output in outputs:
                new_text = output.outputs[0].text
                all_responses.append(new_text)
            
            return all_responses
            
        except Exception as e:
            print(f"Error in batch_generate_next_layer: {e}")
            return [""] * len(partial_responses)
    
    def check_branch_termination(self, text, initial_length):
        """Check if branch should terminate"""
        if len(text) <= initial_length:
            return False, "none"
        
        new_content = text[initial_length:]
        
        try:
            if text.endswith(self.tokenizer.eos_token):
                return True, "eos_token"
        except:
            pass
        
        return False, "none"
    
    def verify_answer_correctness(self, text, branch_path, solution):
        """Verify correctness of branch answer"""
        try:
            result = compute_score(
                solution_str=text,
                ground_truth=solution,
                strict_box_verify=self.strict_box_verify
            )
            
            is_correct = result['acc']
            extracted_answer = result['pred']
            
            if is_correct:
                status = "CORRECT"
            else:
                status = f"WRONG (got: {extracted_answer}, expected: {solution})"
                
            return bool(is_correct), extracted_answer, status
            
        except Exception as e:
            error_msg = f"VERIFICATION ERROR: {e}"
            return False, None, error_msg
    
    def extract_boxed_answer(self, text, initial_length):
        """Extract \\boxed{} answer from text"""
        try:
            new_content = text[initial_length:] if len(text) > initial_length else text
            
            if "\\boxed{" in new_content:
                import re
                pattern = r'\\boxed\{([^}]*)\}'
                matches = re.findall(pattern, new_content)
                if matches:
                    return matches[-1]
            
            return "Invalid"
        except Exception as e:
            return "Invalid"
    
    def add_to_global_rollout_queue(self, text, branch_path, question_id, solution):
        """Add to global rollout queue"""
        if not hasattr(self, 'global_rollout_queue'):
            self.global_rollout_queue = []
        
        self.global_rollout_queue.append((text, branch_path, question_id, solution))
    
    def process_global_rollout_queue(self, question_states):
        """Process global rollout queue"""
        if not hasattr(self, 'global_rollout_queue') or not self.global_rollout_queue:
            return
        
        print(f"Processing global rollout queue with {len(self.global_rollout_queue)} branches...")
        
        # Collect all texts needing rollout
        all_rollout_texts = []
        rollout_info = []  # [(question_id, branch_path, solution), ...]
        
        for text, branch_path, question_id, solution in self.global_rollout_queue:
            all_rollout_texts.extend([text] * self.rollout_times)
            rollout_info.extend([(question_id, branch_path, solution)] * self.rollout_times)
        
        print(f"Batch rollout generating {len(all_rollout_texts)} responses with 3072 tokens each...")
        
        all_rollout_results = self.batch_rollout_vllm(all_rollout_texts)
        
        # Process results grouped by question
        for i, (text, branch_path, question_id, solution) in enumerate(self.global_rollout_queue):
            start_idx = i * self.rollout_times
            end_idx = start_idx + self.rollout_times
            branch_rollout_results = all_rollout_results[start_idx:end_idx]
            
            validated_results = []
            for rollout_text, _ in branch_rollout_results:
                try:
                    result = compute_score(
                        solution_str=text + rollout_text,
                        ground_truth=solution,
                        strict_box_verify=self.strict_box_verify
                    )
                    is_correct = bool(result['acc'])
                except Exception as e:
                    is_correct = False
                validated_results.append((rollout_text, is_correct))
            
            correct_count = sum(1 for _, is_correct in validated_results if is_correct)
            rollout_acc = correct_count / self.rollout_times if self.rollout_times > 0 else 0.0
            
            # Save to corresponding question's results
            question_states[question_id]['branch_results'][branch_path] = {
                'boxed_answer': '[ROLLOUT]',
                'extracted_answer': '[ROLLOUT]',
                'is_correct': rollout_acc > 0,
                'termination_reason': f'rollout_{self.rollout_times}',
                'status': f'ROLLOUT ACCURACY: {rollout_acc:.2f} ({int(rollout_acc*self.rollout_times)}/{self.rollout_times})',
                'rollout_results': validated_results
            }
        
        self.global_rollout_queue.clear()
    
    def batch_rollout_vllm(self, rollout_texts):
        """Batch rollout"""
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                n=1,
                temperature=0.7,
                top_p=1.0,
                max_tokens=3072
            )
            
            outputs = self.llm.generate(rollout_texts, sampling_params)
            
            all_results = []
            for output in outputs:
                new_content = output.outputs[0].text
                # Note: Cannot directly verify here because we need the corresponding solution
                # Verification will be handled in process_global_rollout_queue
                all_results.append((new_content.strip(), False))  # Temporarily set to False
            
            return all_results
            
        except Exception as e:
            print(f"Error in batch_rollout_vllm: {e}")
            return [("ERROR", False)] * len(rollout_texts)
    
    def save_question_results(self, question_id, state):
        """Save results for a single question"""
        with open(state['save_path'], 'w', encoding='utf-8') as f:
            f.write("Batch Token-Level Branching Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Question: {state['question_text']}\n")
            f.write(f"Solution: {state['solution']}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(state['log_content']))
        
        overview_path = state['save_path'].replace('.txt', '_overview.txt')
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write("K-ary Tree Overview - Batch Processing\n")
            f.write("=" * 60 + "\n")
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Ground Truth: {state['solution']}\n")
            f.write(f"Total Branches: {len(state['branch_results'])}\n")
            f.write("=" * 60 + "\n\n")
            
            correct_count = sum(1 for result in state['branch_results'].values() if result['is_correct'])
            total_count = len(state['branch_results'])
            
            f.write(f"Summary: {correct_count}/{total_count} branches correct\n")
            f.write(f"Final Accuracy: {correct_count/total_count*100:.2f}%\n\n")
            
            for branch_path, result in sorted(state['branch_results'].items()):
                status_icon = "✅" if result['is_correct'] else "❌"
                f.write(f"{branch_path}: {status_icon} {result['boxed_answer']} - {result['termination_reason']}\n")
        
        if self.global_file_manager:
            log_content = "\n".join(state['log_content'])
            self.global_file_manager.add_question_result(
                question_id=question_id,
                question_text=state['question_text'],
                solution=state['solution'],
                branch_results=state['branch_results'],
                log_content=log_content
            )