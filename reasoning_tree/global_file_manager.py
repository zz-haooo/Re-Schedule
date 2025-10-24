import os
import datetime

class GlobalFileManager:
    """Global file manager for optimizing file generation"""
    
    def __init__(self, model_name, result_dir):
        self.model_name = model_name
        self.result_dir = result_dir
        self.all_results = []
        self.all_overviews = []
        
        # Create two main file paths
        self.main_log_path = os.path.join(result_dir, f"{model_name}_all_results.txt")
        self.main_overview_path = os.path.join(result_dir, f"{model_name}_all_overview.txt")
        
        # Initialize main files
        self._init_main_files()
    
    def _init_main_files(self):
        """Initialize main files"""
        with open(self.main_log_path, 'w', encoding='utf-8') as f:
            f.write("Global Token-Level Branching Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        with open(self.main_overview_path, 'w', encoding='utf-8') as f:
            f.write("Global K-ary Tree Overview - All Questions Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def add_question_result(self, question_id, question_text, solution, branch_results, log_content):
        """Add result for a single question"""
        # Add to result list
        self.all_results.append({
            'question_id': question_id,
            'question_text': question_text,
            'solution': solution,
            'branch_results': branch_results,
            'log_content': log_content
        })
        
        # Write to main log file
        with open(self.main_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Question {question_id}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Question: {question_text}\n")
            f.write(f"Solution: {solution}\n")
            f.write(f"{'='*60}\n")
            f.write(log_content)
            f.write(f"\n{'='*60}\n")
        
        # Write to main overview file
        with open(self.main_overview_path, 'a', encoding='utf-8') as f:
            f.write(f"\nQuestion {question_id}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Question: {question_text[:100]}...\n")
            f.write(f"Solution: {solution}\n")
            
            # Count branch results
            total_branches = len(branch_results)
            correct_branches = sum(1 for result in branch_results.values() if result['is_correct'])
            accuracy = correct_branches / total_branches if total_branches > 0 else 0.0
            
            f.write(f"Total Branches: {total_branches}\n")
            f.write(f"Correct Branches: {correct_branches}\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            
            # Show first few branch results
            sorted_branches = sorted(branch_results.items(), key=lambda x: self._sort_branch_key(x[0]))
            for i, (branch_path, result) in enumerate(sorted_branches[:5]):  # Show only first 5
                status_icon = "✅" if result['is_correct'] else "❌"
                f.write(f"  {branch_path}: {status_icon} {result['boxed_answer']}\n")
            
            if len(sorted_branches) > 5:
                f.write(f"  ... and {len(sorted_branches) - 5} more branches\n")
            f.write(f"{'-'*40}\n")
    
    def _sort_branch_key(self, branch_path):
        """Generate sort key for branch path"""
        if branch_path == 'ROOT':
            return (0,)
        try:
            parts = [int(part) for part in branch_path.split('.')]
            return tuple(parts)
        except:
            return (999, 999, 999)
    
    def finalize_results(self):
        """Finalize all results and generate summary"""
        # Calculate overall statistics
        total_questions = len(self.all_results)
        total_accuracy = 0.0
        total_branches = 0
        total_correct_branches = 0
        
        for result in self.all_results:
            branch_results = result['branch_results']
            question_branches = len(branch_results)
            question_correct = sum(1 for r in branch_results.values() if r['is_correct'])
            
            total_branches += question_branches
            total_correct_branches += question_correct
            
            if question_branches > 0:
                total_accuracy += question_correct / question_branches
        
        overall_accuracy = total_accuracy / total_questions if total_questions > 0 else 0.0
        
        # Write summary to main overview file
        with open(self.main_overview_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"GLOBAL SUMMARY\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Total Branches: {total_branches}\n")
            f.write(f"Total Correct Branches: {total_correct_branches}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
            f.write(f"Average Branches per Question: {total_branches/total_questions:.1f}\n")
            f.write(f"{'='*80}\n")
        
        print(f"Global results saved to:")
        print(f"   - Main log: {self.main_log_path}")
        print(f"   - Main overview: {self.main_overview_path}")
        
        return {
            'total_questions': total_questions,
            'total_branches': total_branches,
            'total_correct_branches': total_correct_branches,
            'overall_accuracy': overall_accuracy
        }