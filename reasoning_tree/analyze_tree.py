"""
K-ary tree analysis program
Analyze progressive_reasoning_overview.txt files, calculate accuracy improvement metrics
Only supports pruning_improvement_topM_sum and parent_selection_improvement_topM_sum
"""

import re
import os
import math
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

class TreeNode:
    def __init__(self, path: str):
        self.path = path
        self.is_correct = None
        self.children = []
        self.parent = None
        self.is_leaf = True
        self.depth = 0  # Node depth (layer number), root node is 0
        
    def __repr__(self):
        return f"TreeNode({self.path}, correct={self.is_correct}, children={len(self.children)}, depth={self.depth})"

class KAryTreeAnalyzer:
    def __init__(self):
        self.nodes = {}
        self.root_nodes = []
        
    def parse_overview_file(self, file_path: str) -> Tuple[int, int, int]:
        """Parse overview file, return (ground_truth, total_branches, correct_branches)"""
        ground_truth = None
        total_branches = 0
        correct_branches = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract Ground Truth
        gt_match = re.search(r'Ground Truth:\s*([-\d]+)', content)
        if gt_match:
            ground_truth = int(gt_match.group(1))
            
        # Extract Total Branches
        tb_match = re.search(r'Total Branches:\s*(\d+)', content)
        if tb_match:
            total_branches = int(tb_match.group(1))
            
        # Extract correct branch count (e.g.: "171/256 branches correct")
        cb_match = re.search(r'(\d+)/(\d+)\s+branches\s+correct', content)
        if cb_match:
            correct_branches = int(cb_match.group(1))
            
        # Check file format type
        if "Batch Processing" in content:
            # New format: X.X.X.X.X: checkmark/cross [ROLLOUT]
            self._parse_new_format(content)
        else:
            # Old format: Branch X.X.X: Answer: Result: Termination:
            self._parse_old_format(content)
            
        # Build tree structure
        self._build_tree_structure()
        
        return ground_truth, total_branches, correct_branches
    
    def _get_node_correctness_counts(self, node: TreeNode) -> Tuple[int, int]:
        """Get the count of correct and incorrect leaf nodes in node's subtree"""
        if node.is_leaf:
            # Leaf node returns its own state directly
            if node.is_correct is True:
                return 1, 0
            elif node.is_correct is False:
                return 0, 1
            else:
                return 0, 0
        
        # Non-leaf node counts all leaf nodes in subtrees
        total_correct = 0
        total_incorrect = 0
        
        for child in node.children:
            correct, incorrect = self._get_node_correctness_counts(child)
            total_correct += correct
            total_incorrect += incorrect
        
        return total_correct, total_incorrect

    def calculate_pruning_accuracy_improvement_topM_sum(self, top_m: int = 5) -> float:
        """
        Calculate the sum of accuracy improvements from M iterations of pruning.
        
        Iterative pruning logic:
        1. Find the node with maximum accuracy improvement after pruning on current tree
        2. Actually prune that node (remove the node and all its descendants from tree)
        3. Repeat steps 1-2 on new tree, M times total
        4. Accumulate accuracy improvements from M prunings
        
        Concept clarification:
        - Real leaf nodes: Deepest layer rollout results, used for accuracy calculation
        - Prunable nodes: Internal nodes excluding root, real leaf nodes and their parents
        
        Args:
            top_m: Number of pruning iterations
        
        Returns:
            Sum of accuracy improvements from M iterations of pruning
        """
        # Deep copy current tree state to avoid modifying original tree
        working_tree = self._create_tree_copy()
        
        total_improvement = 0.0
        current_accuracy = working_tree._calculate_overall_accuracy()
        
        for iteration in range(top_m):
            # Find node with maximum accuracy improvement after pruning in current tree
            best_node, best_improvement = working_tree._find_best_pruning_node()
            
            if best_node is None or best_improvement <= 0:
                # No more prunable nodes or no positive improvement, stop iteration
                break
            
            working_tree._prune_node_permanently(best_node)
            
            # Accumulate accuracy improvement
            total_improvement += best_improvement
            
            current_accuracy += best_improvement
        
        return float(total_improvement)
    
    def calculate_parent_child_selection_improvement_topM_sum(self, top_m: int = 5) -> float:
        """
        Calculate the sum of accuracy improvements from parent selecting best child M times.
        
        Logic explanation:
        1. Find the case with maximum accuracy improvement when parent selects one child and removes others
        2. Actually perform the selection (remove other children from tree, keep only the highest accuracy child)
        3. Repeat steps 1-2 on new tree, M times total
        4. Accumulate accuracy improvements from M selections
        
        Concept clarification:
        - Real leaf nodes: Deepest layer rollout results, used for accuracy calculation
        - Selectable nodes: Internal nodes excluding root and real leaf nodes, must have multiple children
        
        Args:
            top_m: Number of selection iterations
        
        Returns:
            Sum of accuracy improvements from M iterations of parent-child selection
        """
        # Deep copy current tree state to avoid modifying original tree
        working_tree = self._create_tree_copy()
        
        total_improvement = 0.0
        current_accuracy = working_tree._calculate_overall_accuracy()
        
        for iteration in range(top_m):
            # Find parent-child selection with maximum accuracy improvement in current tree
            best_parent, best_child, best_improvement = working_tree._find_best_parent_selection_node()
            
            if best_parent is None or best_improvement <= 0:
                # No more selectable nodes or no positive improvement, stop iteration
                break
            
            # Actually perform parent's child selection
            working_tree._perform_parent_selection(best_parent, best_child)
            
            # Accumulate accuracy improvement
            total_improvement += best_improvement
            
            current_accuracy += best_improvement
        
        return float(total_improvement)
    
    def _calculate_overall_accuracy(self) -> float:
        """
        Calculate overall accuracy of current tree
        Based on all real leaf nodes (rollout results): correct rollouts / total rollouts
        """
        all_leaves = []
        for root in self.root_nodes:
            all_leaves.extend(self.get_all_leaf_nodes(root))
        
        if not all_leaves:
            return 0.0
        
        correct_count = sum(1 for leaf in all_leaves if leaf.is_correct is True)
        total_count = len([leaf for leaf in all_leaves if leaf.is_correct is not None])
        
        return (correct_count / total_count) if total_count > 0 else 0.0
    
    def _calculate_accuracy_after_pruning(self, pruning_node: TreeNode) -> float:
        """
        Calculate overall accuracy after pruning specified node
        
        Pruning logic: Remove all real leaf nodes (deepest layer rollout results) under the pruned node,
        then recalculate overall accuracy from remaining real leaf nodes.
        
        Args:
            pruning_node: Node to be pruned
        
        Returns:
            Overall accuracy after pruning
        """
        all_real_leaves = []
        for root in self.root_nodes:
            all_real_leaves.extend(self.get_all_leaf_nodes(root))
        
        if not all_real_leaves:
            return 0.0
        
        pruned_descendants = self.get_all_descendants(pruning_node)
        pruned_descendants.append(pruning_node)  # Include the pruned node itself
        
        # Find the real leaf nodes being pruned
        pruned_real_leaves = [leaf for leaf in all_real_leaves 
                             if leaf in pruned_descendants]
        
        remaining_real_leaves = [leaf for leaf in all_real_leaves 
                               if leaf not in pruned_descendants and leaf.is_correct is not None]
        
        if not remaining_real_leaves:
            return 0.0
        
        correct_count = sum(1 for leaf in remaining_real_leaves if leaf.is_correct is True)
        total_count = len(remaining_real_leaves)
        
        return (correct_count / total_count) if total_count > 0 else 0.0
    
    def _calculate_subtree_accuracy(self, node: TreeNode) -> float:
        """
        Calculate accuracy of specified node's subtree
        
        Args:
            node: Target node
        
        Returns:
            Subtree accuracy (0.0-1.0)
        """
        subtree_leaves = self.get_all_leaf_nodes(node)
        
        if not subtree_leaves:
            return 0.0
        
        correct_count = sum(1 for leaf in subtree_leaves if leaf.is_correct is True)
        total_count = len([leaf for leaf in subtree_leaves if leaf.is_correct is not None])
        
        return (correct_count / total_count) if total_count > 0 else 0.0
    
    def _calculate_accuracy_after_parent_selection(self, parent_node: TreeNode, selected_child: TreeNode) -> float:
        """
        Calculate overall accuracy after parent selects specified child (removing other children)
        
        Logic:
        1. Get all real leaf nodes
        2. Find all real leaf nodes under other removed children
        3. Calculate overall accuracy from remaining real leaf nodes
        
        Args:
            parent_node: Parent node
            selected_child: Selected child to keep
        
        Returns:
            Overall accuracy after selection
        """
        all_real_leaves = []
        for root in self.root_nodes:
            all_real_leaves.extend(self.get_all_leaf_nodes(root))
        
        if not all_real_leaves:
            return 0.0
        
        removed_children = [child for child in parent_node.children if child != selected_child]
        
        removed_real_leaves = []
        for removed_child in removed_children:
            removed_descendants = self.get_all_descendants(removed_child)
            removed_descendants.append(removed_child)  # Include the removed child itself
            
            # Find real leaf nodes being removed
            for leaf in all_real_leaves:
                if leaf in removed_descendants:
                    removed_real_leaves.append(leaf)
        
        remaining_real_leaves = [leaf for leaf in all_real_leaves 
                               if leaf not in removed_real_leaves and leaf.is_correct is not None]
        
        if not remaining_real_leaves:
            return 0.0
        
        correct_count = sum(1 for leaf in remaining_real_leaves if leaf.is_correct is True)
        total_count = len(remaining_real_leaves)
        
        return (correct_count / total_count) if total_count > 0 else 0.0
    
    def _create_tree_copy(self):
        """Create a deep copy of current tree for iterative pruning"""
        # Create new analyzer instance
        copy_analyzer = KAryTreeAnalyzer()
        
        # Copy all nodes
        for path, original_node in self.nodes.items():
            new_node = TreeNode(path)
            new_node.is_correct = original_node.is_correct
            new_node.is_leaf = original_node.is_leaf
            new_node.depth = original_node.depth
            copy_analyzer.nodes[path] = new_node
        
        # Rebuild tree structure (parent-child relationships)
        for path, original_node in self.nodes.items():
            new_node = copy_analyzer.nodes[path]
            
            # Copy child relationships
            for child in original_node.children:
                if child.path in copy_analyzer.nodes:
                    new_child = copy_analyzer.nodes[child.path]
                    new_node.children.append(new_child)
                    new_child.parent = new_node
            
            # Copy parent relationship
            if original_node.parent and original_node.parent.path in copy_analyzer.nodes:
                new_node.parent = copy_analyzer.nodes[original_node.parent.path]
        
        # Copy root node list
        for root in self.root_nodes:
            if root.path in copy_analyzer.nodes:
                copy_analyzer.root_nodes.append(copy_analyzer.nodes[root.path])
        
        return copy_analyzer
    
    def _find_best_pruning_node(self) -> Tuple[Optional[TreeNode], float]:
        """
        Find node with maximum accuracy improvement after pruning in current tree
        
        Returns:
            (best node, maximum improvement) or (None, 0.0) if no prunable nodes
        """
        current_accuracy = self._calculate_overall_accuracy()
        best_node = None
        best_improvement = 0.0
        
        for node in self.nodes.values():
            # Skip root node
            if node.parent is None:
                continue
            # Skip leaf nodes
            if node.is_leaf:
                continue
            # Skip parents of leaf nodes (penultimate layer)
            if all(child.is_leaf for child in node.children):
                continue
            if len(node.children) == 0:
                continue
            
            # Calculate accuracy improvement after pruning this node
            pruned_accuracy = self._calculate_accuracy_after_pruning(node)
            improvement = pruned_accuracy - current_accuracy
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_node = node
        
        return best_node, best_improvement
    
    def _find_best_parent_selection_node(self) -> Tuple[Optional[TreeNode], Optional[TreeNode], float]:
        """
        Find parent-child selection with maximum accuracy improvement in current tree
        
        Logic:
        1. Iterate through all candidate parent nodes (non-root, non-leaf, have multiple children)
        2. For each parent, calculate accuracy improvement from selecting each child
        3. When selecting child, remove all other children of that parent, keep only optimal child
        4. Return maximum improvement (parent, optimal child, improvement)
        
        Returns:
            (best parent, best child, maximum improvement) or (None, None, 0.0) if no selectable nodes
        """
        current_accuracy = self._calculate_overall_accuracy()
        best_parent = None
        best_child = None
        best_improvement = 0.0
        
        for node in self.nodes.values():
            # Skip root node
            if node.parent is None:
                continue
            # Skip leaf nodes
            if node.is_leaf:
                continue
            # Must have multiple children to perform selection
            if len(node.children) <= 1:
                continue
            # Skip parents of leaf nodes (penultimate layer), these nodes' children are all leaves
            if all(child.is_leaf for child in node.children):
                continue
            
            # Calculate subtree accuracy for each child, find optimal child
            best_child_for_this_parent = None
            best_child_accuracy = -1.0
            
            for child in node.children:
                child_accuracy = self._calculate_subtree_accuracy(child)
                if child_accuracy > best_child_accuracy:
                    best_child_accuracy = child_accuracy
                    best_child_for_this_parent = child
            
            if best_child_for_this_parent is None:
                continue
            
            # Calculate overall accuracy improvement after selecting optimal child
            improved_accuracy = self._calculate_accuracy_after_parent_selection(node, best_child_for_this_parent)
            improvement = improved_accuracy - current_accuracy
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_parent = node
                best_child = best_child_for_this_parent
        
        return best_parent, best_child, best_improvement
    
    def _prune_node_permanently(self, node_to_prune: TreeNode):
        """
        Permanently remove specified node and all its descendants from tree
        
        Args:
            node_to_prune: Node to be pruned
        """
        # Get all nodes to be removed (including the node and all its descendants)
        nodes_to_remove = [node_to_prune]
        nodes_to_remove.extend(self.get_all_descendants(node_to_prune))
        
        # Remove node from parent's children list
        if node_to_prune.parent:
            if node_to_prune in node_to_prune.parent.children:
                node_to_prune.parent.children.remove(node_to_prune)
        
        # Remove from root nodes list (if it is a root node)
        if node_to_prune in self.root_nodes:
            self.root_nodes.remove(node_to_prune)
        
        # Remove all related nodes from node dictionary
        for node in nodes_to_remove:
            if node.path in self.nodes:
                del self.nodes[node.path]
    
    def _perform_parent_selection(self, parent_node: TreeNode, selected_child: TreeNode):
        """
        Permanently perform parent's child selection, remove all other children
        
        Logic:
        1. Remove all children of parent_node except selected_child
        2. Delete removed children and all their descendants from node dictionary
        3. Update tree structure
        
        Args:
            parent_node: Parent node
            selected_child: Selected child to keep
        """
        if selected_child not in parent_node.children:
            return  # Safety check
        
        # Get children to remove (all children except selected_child)
        children_to_remove = [child for child in parent_node.children if child != selected_child]
        
        # Remove each child and all its descendants
        for child_to_remove in children_to_remove:
            # Get all nodes to remove (including the child and all its descendants)
            nodes_to_remove = [child_to_remove]
            nodes_to_remove.extend(self.get_all_descendants(child_to_remove))
            
            # Remove child from parent's children list
            if child_to_remove in parent_node.children:
                parent_node.children.remove(child_to_remove)
            
            # Remove all related nodes from node dictionary
            for node in nodes_to_remove:
                if node.path in self.nodes:
                    del self.nodes[node.path]
    
    def _parse_new_format(self, content: str):
        """Parse new format file content"""
        # Find all X.X.X.X.X: checkmark/cross [ROLLOUT] format entries
        # Example: 1.1.1.1.1: checkmark [ROLLOUT] - rollout_1
        branch_pattern = r'^([\d\.]+):\s*([✅❌])\s*\[ROLLOUT\]'
        
        for line in content.split('\n'):
            match = re.search(branch_pattern, line.strip())
            if match:
                branch_path = match.group(1).strip()
                result_symbol = match.group(2)
                
                node = TreeNode(branch_path)
                
                if result_symbol == '✅':
                    node.is_correct = True
                elif result_symbol == '❌':
                    node.is_correct = False
                else:
                    node.is_correct = None
                    
                self.nodes[node.path] = node
    
    def _parse_old_format(self, content: str):
        """Parse old format file content"""
        # Find all Branch entries
        branch_pattern = r'Branch\s+([\d\.]+):\s*\n\s*Answer:\s*([^\n]+)\s*\n\s*Result:\s*([^✅❌]*[✅❌][^✅❌]*)\s*\n\s*Termination:'
        
        branches = re.findall(branch_pattern, content)
        
        for branch_path, answer, result in branches:
            # Create node
            node = TreeNode(branch_path.strip())
            
            # Determine correctness
            if 'CORRECT' in result.upper():
                node.is_correct = True
            elif 'WRONG' in result.upper():
                node.is_correct = False
            else:
                node.is_correct = None
                
            self.nodes[node.path] = node
    
    def _build_tree_structure(self):
        """Build parent-child relationships in tree"""
        # Sort by path length to ensure parent nodes are processed first
        sorted_paths = sorted(self.nodes.keys(), key=lambda x: (len(x.split('.')), x))
        
        for path in sorted_paths:
            node = self.nodes[path]
            path_parts = path.split('.')
            
            node.depth = len(path_parts) - 1
            
            if len(path_parts) == 1:
                self.root_nodes.append(node)
            else:
                parent_path = '.'.join(path_parts[:-1])
                if parent_path in self.nodes:
                    parent = self.nodes[parent_path]
                    parent.children.append(node)
                    parent.is_leaf = False
                    node.parent = parent
                else:
                    # If parent doesn't exist, create it
                    parent = TreeNode(parent_path)
                    parent.is_leaf = False
                    parent.depth = len(parent_path.split('.')) - 1
                    self.nodes[parent_path] = parent
                    parent.children.append(node)
                    node.parent = parent
                    
                    # Recursively process higher-level parent nodes
                    self._create_missing_ancestors(parent_path)
    
    def _create_missing_ancestors(self, path: str):
        """Recursively create missing ancestor nodes"""
        path_parts = path.split('.')
        if len(path_parts) <= 1:
            if path not in [r.path for r in self.root_nodes]:
                self.root_nodes.append(self.nodes[path])
            return
            
        parent_path = '.'.join(path_parts[:-1])
        if parent_path not in self.nodes:
            parent = TreeNode(parent_path)
            parent.is_leaf = False
            parent.depth = len(parent_path.split('.')) - 1  # Set depth
            self.nodes[parent_path] = parent
            self._create_missing_ancestors(parent_path)
            
        # Establish parent-child relationship
        if parent_path in self.nodes:
            parent = self.nodes[parent_path]
            current = self.nodes[path]
            if current not in parent.children:
                parent.children.append(current)
                current.parent = parent
    
    def get_all_leaf_nodes(self, node: TreeNode) -> List[TreeNode]:
        """Get all leaf nodes under a given node"""
        if node.is_leaf:
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaf_nodes(child))
        return leaves
    
    def get_all_descendants(self, node: TreeNode) -> List[TreeNode]:
        """Get all descendant nodes under a given node (including direct children and deeper level nodes)"""
        descendants = []
        for child in node.children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))
        return descendants
    
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze single file"""
        print(f"\nAnalyzing file: {file_path}")
        
        ground_truth, total_branches, correct_branches = self.parse_overview_file(file_path)
        
        final_accuracy = (correct_branches / total_branches * 100) if total_branches > 0 else 0.0
        
        print(f"Ground Truth: {ground_truth}")
        print(f"Total Branches: {total_branches}")
        print(f"Correct Branches: {correct_branches}")
        print(f"Final Accuracy: {final_accuracy:.2f}%")
        print(f"Parsed node count: {len(self.nodes)}")
        
        return {
            'file_path': file_path,
            'ground_truth': ground_truth,
            'total_branches': total_branches,
            'correct_branches': correct_branches,
            'final_accuracy': final_accuracy,
            'total_nodes': len(self.nodes),
        }

def analyze_directory(directory_path: str):
    """Analyze all overview files in directory"""
    results = []
    
    # Find all overview files
    overview_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_overview.txt'):
                overview_files.append(os.path.join(root, file))
    
    print(f"Found {len(overview_files)} overview files")
    
    for file_path in sorted(overview_files):
        analyzer = KAryTreeAnalyzer()
        try:
            result = analyzer.analyze_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            continue
    
    # Summary statistics
    print(f"\nSummary statistics:")
    print(f"  Total files: {len(results)}")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze K-ary tree structure for pruning and parent-selection improvement metrics')
    parser.add_argument('path', nargs='?', help='Overview file path or directory path containing overview files')
    parser.add_argument('--output', '-o', help='Output results to JSON file')
    parser.add_argument('--metric', choices=['pruning_improvement_topM_sum', 'parent_selection_improvement_topM_sum'], 
                       default='parent_selection_improvement_topM_sum', help='Metric type to calculate')
    parser.add_argument('--topM', type=int, default=5, help='M value for top-M calculation')
    
    args = parser.parse_args()
    
    # If no path argument provided, show help
    if not args.path:
        parser.print_help()
        return
    
    if os.path.isfile(args.path):
        analyzer = KAryTreeAnalyzer()
        result = analyzer.analyze_file(args.path)
        
        if args.metric == 'pruning_improvement_topM_sum':
            metric_value = analyzer.calculate_pruning_accuracy_improvement_topM_sum(top_m=args.topM)
            print(f"\nPruning improvement (top-{args.topM}): {metric_value:.6f}")
        else:
            metric_value = analyzer.calculate_parent_child_selection_improvement_topM_sum(top_m=args.topM)
            print(f"\nParent selection improvement (top-{args.topM}): {metric_value:.6f}")
        
        results = [result]
    elif os.path.isdir(args.path):
        results = analyze_directory(args.path)
    else:
        print(f"Path does not exist: {args.path}")
        return
    
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()