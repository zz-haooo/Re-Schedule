import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import json
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import datetime
import argparse
import numpy as np
from collections import defaultdict
import sys
from functools import lru_cache
import torch.nn.functional as F

from batch_processor import BatchProcessor
from global_file_manager import GlobalFileManager

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

parser = argparse.ArgumentParser(description='Batch Parallel Token-Level Branching Evaluation')

parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
parser.add_argument("--log_dir", type=str, default="eval_logs", help="Log directory")
parser.add_argument("--result_dir", type=str, help="Result directory")
parser.add_argument("--dataset_name", type=str, default="dapo_converted", help="Dataset name")
parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split")
parser.add_argument("--math_subset", type=str, default="algebra", help="Math subset")

parser.add_argument("--branch_count", type=int, default=3, help="Number of branches at each branching point")
parser.add_argument("--timestamp", type=str, help="Custom timestamp for result folder")
parser.add_argument("--num_questions", type=int, default=3, help="Number of questions to process")
parser.add_argument("--max_branch_depth", type=int, default=5, help="Maximum branch depth")
parser.add_argument("--rollout_times", type=int, default=1, help="Number of rollouts at max branch depth")
parser.add_argument("--heuristic_branch_interval", type=int, default=100, help="Interval for heuristic branching")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for parallel processing")

parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling threshold")
parser.add_argument("--max_tokens", type=int, default=4096, help="Max generated tokens")
parser.add_argument("--strict_box", action="store_true", default=True, help="Use strict box verification")

parser.add_argument("--gpu_devices", type=str, default="0,1,2,3", help="GPU devices to use (comma-separated)")
parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size")

parser.add_argument("--start_index", type=int, default=None, help="Start index for sample range (0-based)")
parser.add_argument("--end_index", type=int, default=None, help="End index for sample range (exclusive, 0-based)")

args = parser.parse_args()

MODEL_PATH = args.model_path
LOG_DIR = args.log_dir
DATASET_NAME = args.dataset_name
DATASET_SPLIT = args.dataset_split
MATH_SUBSET = args.math_subset
STRICT_BOX_VERIFY = args.strict_box

BRANCH_COUNT = args.branch_count
BATCH_SIZE = args.batch_size

GPU_DEVICES = args.gpu_devices
TENSOR_PARALLEL_SIZE = args.tensor_parallel_size

START_INDEX = args.start_index
END_INDEX = args.end_index

MODEL_NAME = MODEL_PATH.split("/")[-1]

# Use external timestamp if provided, otherwise generate new one
if args.timestamp:
    TIMESTAMP = args.timestamp
else:
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_DIR = args.result_dir if args.result_dir else os.path.join(LOG_DIR, f"batch_results_{TIMESTAMP}")

# Ensure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading model with VLLM...")
print(f"   - GPU Devices: {GPU_DEVICES}")
print(f"   - Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES

from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.95, 
    max_num_seqs=256,
    enforce_eager=False
)
model = None
print(f"VLLM model loaded successfully!")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def get_simple_prompt(question):
    """Create simple prompt containing only system prompt and question"""
    if not isinstance(question, str):
        question = str(question)
    
    prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]
    return prompt

def extract_question_text(question_data):
    """Extract actual question text from complex question format"""
    try:
        if isinstance(question_data, str):
            if question_data.startswith("[{"):
                import ast
                try:
                    parsed = ast.literal_eval(question_data)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        content = parsed[0].get('content', '')
                        content = content.replace('Solve the following math problem step by step. The last line of your response should be of the form \\\\boxed{Answer} where Answer is the answer to the problem.\\n\\n', '')
                        content = content.replace('\\n\\nRemember to put your answer in \\\\boxed{...} format on its own line.', '')
                        return content.strip()
                except:
                    pass
            return question_data
        elif isinstance(question_data, list) and len(question_data) > 0:
            content = question_data[0].get('content', '')
            content = content.replace('Solve the following math problem step by step. The last line of your response should be of the form \\\\boxed{Answer} where Answer is the answer to the problem.\\n\\n', '')
            content = content.replace('\\n\\nRemember to put your answer in \\\\boxed{...} format on its own line.', '')
            return content.strip()
        else:
            return str(question_data)
    except Exception as e:
        return str(question_data)

def get_dapo_converted_questions(split="train", name='all') -> Dataset:
    """Load DAPO-Math-17k-converted dataset with the new format"""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_path = os.path.join(project_root, 'datasets', 'DAPO-Math-17k.parquet')
    
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        data = Dataset.from_pandas(df)
        
        data = data.map(lambda x: {
            'prompt': get_simple_prompt(x['prompt'][0]['content']),
            'solution': x['reward_model']['ground_truth'],
            'question': x['prompt'][0]['content'],
            'clean_question': extract_question_text(x['prompt'][0]['content'])
        })
        
        print(f"Successfully loaded DAPO-Math-17k-converted dataset from {parquet_path}")
        print(f"   - Total samples: {len(data)}")
        print(f"   - First question preview: {data[0]['question'][:100]}...")
        
        return data
        
    except Exception as e:
        print(f"Error loading DAPO-Math-17k-converted dataset: {e}")
        # Fallback to datasets directory
        path = os.path.join(project_root, 'datasets', 'DAPO-Math-17k-converted')
        data = load_dataset(path, name)[split]
        data = data.map(lambda x: {
            'prompt': get_simple_prompt(x['prompt'][0]['content']),
            'solution': x['reward_model']['ground_truth'],
            'question': x['prompt'][0]['content'],
            'clean_question': extract_question_text(x['prompt'][0]['content'])
        })
        return data

def get_gsm8k_questions(split="train") -> Dataset:
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'datasets', 'gsm8k')

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")
    
    data = load_dataset(file_path, 'default')[split]
    data = data.map(lambda x: {
        'prompt': get_simple_prompt(x['question']),
        'solution': f"{extract_hash_answer(x['answer'])}",
        'question': x['question'],
        'clean_question': extract_question_text(x['question'])
    })
    return data

def get_math_questions(split="train", name='algebra') -> Dataset:
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'datasets', 'MATH')
    
    data = load_dataset(file_path, name)[split]
    data = data.map(lambda x: {
        'prompt': get_simple_prompt(x['problem']),
        'solution': x['solution'],
        'question': x['problem'],
        'clean_question': extract_question_text(x['problem'])
    })
    return data

# Main evaluation function: Batch parallel token-level branching inference
def evaluate_model_with_batch_branching():
    # Get number of questions to process
    NUM_QUESTIONS = args.num_questions
    
    # Load dataset
    print("Loading dataset...")
    if DATASET_NAME == "gsm8k":
        dataset = get_gsm8k_questions(split=DATASET_SPLIT)
    elif DATASET_NAME == "math":
        dataset = get_math_questions(split=DATASET_SPLIT, name=MATH_SUBSET)
    elif DATASET_NAME == "dapo_converted":
        dataset = get_dapo_converted_questions(split=DATASET_SPLIT)
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")
    
    print(f"Dataset loaded. Size: {len(dataset)}")
    print(f"Batch Parallel Token-Level Branching Parameters:")
    print(f"   - Branch Count (K): {BRANCH_COUNT}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Max Branch Depth: {args.max_branch_depth}")
    print(f"   - Rollout Times: {args.rollout_times}")
    print(f"   - Heuristic Branch Interval: {args.heuristic_branch_interval}")
    print(f"   - Number of Questions: {NUM_QUESTIONS}")
    print(f"   - Using VLLM: True")
    
    # Process sample range
    if START_INDEX is not None or END_INDEX is not None:
        # Set default values
        start_idx = START_INDEX if START_INDEX is not None else 0
        end_idx = END_INDEX if END_INDEX is not None else len(dataset)
        
        # Validate range
        if start_idx < 0 or end_idx > len(dataset) or start_idx >= end_idx:
            raise ValueError(f"Invalid sample range: start_index={start_idx}, end_index={end_idx}, dataset_size={len(dataset)}")
        
        actual_num_questions = end_idx - start_idx
        print(f"Processing sample range: {start_idx}-{end_idx-1} ({actual_num_questions} questions)")
        print(f"   - Sample range: [{start_idx}, {end_idx})")
    else:
        # Default to processing first N questions
        actual_num_questions = min(NUM_QUESTIONS, len(dataset))
        start_idx = 0
        end_idx = actual_num_questions
        print(f"\nProcessing {actual_num_questions} questions in batches of {BATCH_SIZE}:")
    
    # Initialize global file manager
    global_file_manager = GlobalFileManager(MODEL_NAME, RESULT_DIR)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        model=model,
        tokenizer=tokenizer,
        llm=llm,
        branch_count=BRANCH_COUNT,
        max_branch_depth=args.max_branch_depth,
        rollout_times=args.rollout_times,
        heuristic_branch_interval=args.heuristic_branch_interval,
        global_file_manager=global_file_manager,
        batch_size=BATCH_SIZE,
        strict_box_verify=STRICT_BOX_VERIFY
    )
    
    all_results = []
    total_accuracy = 0.0
    
    # Process questions in batches
    for batch_start in range(start_idx, end_idx, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, end_idx)
        current_batch_size = batch_end - batch_start
        
        print(f"\n" + "="*60)
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}: questions {batch_start+1}-{batch_end}")
        print("="*60)
        
        # Prepare current batch data
        batch_data = []
        for i in range(batch_start, batch_end):
            item = dataset[i]
            batch_data.append({
                'question_id': i + 1,  # Keep original index
                'question': item['question'],
                'solution': item['solution'],
                'prompt': item['prompt']
            })
        
        # Batch process current batch
        processed_states = batch_processor.process_batch(batch_data, RESULT_DIR, MODEL_NAME)
        
        # Process results for each question
        for question_id, state in processed_states.items():
            final_text = ""
            has_final_answer = False
            
            if state['current_layer']:
                final_text = state['current_layer'][-1][0][state['initial_length']:].strip()
                has_final_answer = True
            
            # Verify correctness of final answer
            final_accuracy = 0.0
            extracted_answer = None
            if has_final_answer:
                try:
                    result = compute_score(
                        solution_str=final_text,
                        ground_truth=state['solution'],
                        strict_box_verify=STRICT_BOX_VERIFY
                    )
                    final_accuracy = float(result['acc'])
                    extracted_answer = result['pred']
                    print(f"   - Question {question_id} final answer correct: {result['acc']}")
                    print(f"   - Extracted answer: {extracted_answer}")
                    print(f"   - Ground truth: {state['solution']}")
                except Exception as e:
                    print(f"   - Question {question_id} answer verification error: {e}")
            
            print(f"   - Question {question_id} accuracy: {final_accuracy*100:.2f}%")
            
            # Accumulate total accuracy
            total_accuracy += final_accuracy
            
            # Save detailed results for current question
            question_result = {
                "question_id": question_id,
                "question": state['question_text'],
                "ground_truth": state['solution'],
                "branching_results": {
                    "total_nodes": len(state['current_layer']),
                    "has_final_answer": has_final_answer,
                    "final_accuracy": final_accuracy,
                    "extracted_answer": extracted_answer,
                    "log_file": f"{MODEL_NAME}_{question_id}.txt",
                    "overview_file": f"{MODEL_NAME}_{question_id}_overview.txt"
                },
                "step_history": []
            }
            
            all_results.append(question_result)
    
    # Calculate overall accuracy
    overall_accuracy = total_accuracy / actual_num_questions if actual_num_questions > 0 else 0.0
    
    print(f"\n" + "="*60)
    print(f"All {actual_num_questions} questions completed!")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}% ({int(total_accuracy)}/{actual_num_questions})")
    print("="*60)
    
    # Finalize global file manager
    global_stats = global_file_manager.finalize_results()
    
    # Save summary results
    summary_results = {
        "timestamp": TIMESTAMP,
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "dataset": DATASET_NAME,
        "num_questions_processed": actual_num_questions,
        "sample_range": {
            "start_index": start_idx,
            "end_index": end_idx,
            "range_size": actual_num_questions
        },
        "batch_parameters": {
            "batch_size": BATCH_SIZE,
            "branch_count": BRANCH_COUNT,
            "max_branch_depth": args.max_branch_depth,
            "rollout_times": args.rollout_times,
            "heuristic_branch_interval": args.heuristic_branch_interval
        },
        "overall_results": {
            "total_accuracy": total_accuracy,
            "overall_accuracy": overall_accuracy,
            "questions_processed": actual_num_questions,
            "global_stats": global_stats
        },
        "individual_results": all_results
    }
    
    # Save summary results to JSON file
    summary_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print(f"Summary results saved to: {summary_path}")
    
    return overall_accuracy

if __name__ == "__main__":
    print("=" * 60)
    print("Batch Parallel Token-Level Branching Evaluation")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"     Path: {MODEL_PATH}")
    print(f"  Dataset: {DATASET_NAME}/{DATASET_SPLIT}")
    print(f"  Batch Parameters:")
    print(f"     - Batch Size: {BATCH_SIZE}")
    print(f"     - Branch Count (K): {BRANCH_COUNT}")
    print(f"     - Max Branch Depth: {args.max_branch_depth}")
    print(f"     - Rollout Times: {args.rollout_times}")
    print(f"     - Heuristic Branch Interval: {args.heuristic_branch_interval}")
    print(f"     - Number of Questions: {args.num_questions}")
    if START_INDEX is not None or END_INDEX is not None:
        start_display = START_INDEX if START_INDEX is not None else 0
        end_display = END_INDEX if END_INDEX is not None else "end"
        print(f"     - Sample Range: [{start_display}, {end_display})")
    print(f"  Model Config:")
    print(f"     - Model Type: VLLM")
    print(f"     - Temperature: {args.temperature}")
    print(f"     - Top-p: {args.top_p}")
    print(f"  Output Directory: {RESULT_DIR}")
    print("=" * 60)
    
    try:
        final_accuracy = evaluate_model_with_batch_branching()
        
        print("\n" + "=" * 60)
        print("Batch Parallel Token-Level Branching Evaluation Complete!")
        print("=" * 60)
        print(f"  Overall Accuracy: {final_accuracy*100:.2f}%")
        print(f"  Results saved to: {RESULT_DIR}")
        print(f"  Generated Files:")
        print(f"     - {MODEL_NAME}_all_results.txt (Main log)")
        print(f"     - {MODEL_NAME}_all_overview.txt (Main overview)")
        print(f"     - {MODEL_NAME}_batch_summary.json (Summary)")
        print(f"     - Individual question files")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print(f"Partial results may be saved in: {RESULT_DIR}") 