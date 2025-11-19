"""
Knowledge Graph for Azuma AI
Tracks relationships between topics, visualizes learning progress
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import networkx as nx
from collections import defaultdict


class KnowledgeNode:
    """Represents a single knowledge node (topic/concept)"""

    def __init__(self, node_id: str, title: str, category: str, difficulty: str):
        self.node_id = node_id
        self.title = title
        self.category = category
        self.difficulty = difficulty
        self.mastery_score = 0.0
        self.unlocked = False
        self.last_practiced = None
        self.prerequisites: Set[str] = set()
        self.dependents: Set[str] = set()

    def update_mastery(self, new_score: float):
        """Update mastery score"""
        self.mastery_score = max(self.mastery_score, new_score)
        self.last_practiced = datetime.now()

    def is_ready_to_learn(self, graph: 'KnowledgeGraph') -> bool:
        """Check if prerequisites are met"""
        if not self.prerequisites:
            return True

        for prereq_id in self.prerequisites:
            prereq_node = graph.get_node(prereq_id)
            if not prereq_node or prereq_node.mastery_score < 0.7:
                return False

        return True


class KnowledgeGraph:
    """
    Manages the knowledge graph for AI/ML learning
    Tracks topic relationships, prerequisites, and learning progress
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self._initialize_ml_knowledge_graph()

    def _initialize_ml_knowledge_graph(self):
        """Initialize the AI/ML knowledge graph structure"""

        # Add fundamental nodes
        self._add_knowledge_node("python_basics", "Python Basics", "fundamentals", "beginner")
        self._add_knowledge_node("numpy", "NumPy", "fundamentals", "beginner")
        self._add_knowledge_node("pandas", "Pandas", "fundamentals", "beginner")
        self._add_knowledge_node("matplotlib", "Matplotlib & Visualization", "fundamentals", "beginner")

        # Math foundations
        self._add_knowledge_node("linear_algebra", "Linear Algebra", "math", "intermediate")
        self._add_knowledge_node("calculus", "Calculus", "math", "intermediate")
        self._add_knowledge_node("probability", "Probability", "math", "intermediate")
        self._add_knowledge_node("statistics", "Statistics", "math", "intermediate")

        # ML Basics
        self._add_knowledge_node("ml_intro", "Introduction to ML", "machine_learning", "beginner")
        self._add_knowledge_node("supervised_learning", "Supervised Learning", "machine_learning", "beginner")
        self._add_knowledge_node("unsupervised_learning", "Unsupervised Learning", "machine_learning", "intermediate")
        self._add_knowledge_node("linear_regression", "Linear Regression", "machine_learning", "beginner")
        self._add_knowledge_node("logistic_regression", "Logistic Regression", "machine_learning", "beginner")
        self._add_knowledge_node("decision_trees", "Decision Trees", "machine_learning", "intermediate")
        self._add_knowledge_node("random_forests", "Random Forests", "machine_learning", "intermediate")
        self._add_knowledge_node("svm", "Support Vector Machines", "machine_learning", "intermediate")
        self._add_knowledge_node("clustering", "Clustering", "machine_learning", "intermediate")
        self._add_knowledge_node("dimensionality_reduction", "Dimensionality Reduction", "machine_learning", "intermediate")

        # Deep Learning
        self._add_knowledge_node("neural_networks", "Neural Networks", "deep_learning", "intermediate")
        self._add_knowledge_node("backpropagation", "Backpropagation", "deep_learning", "intermediate")
        self._add_knowledge_node("activation_functions", "Activation Functions", "deep_learning", "intermediate")
        self._add_knowledge_node("optimization", "Optimization Algorithms", "deep_learning", "intermediate")
        self._add_knowledge_node("regularization", "Regularization", "deep_learning", "intermediate")
        self._add_knowledge_node("batch_norm", "Batch Normalization", "deep_learning", "advanced")
        self._add_knowledge_node("dropout", "Dropout", "deep_learning", "advanced")

        # Computer Vision
        self._add_knowledge_node("image_processing", "Image Processing", "computer_vision", "intermediate")
        self._add_knowledge_node("cnns", "Convolutional Neural Networks", "computer_vision", "intermediate")
        self._add_knowledge_node("conv_layers", "Convolutional Layers", "computer_vision", "intermediate")
        self._add_knowledge_node("pooling", "Pooling Layers", "computer_vision", "intermediate")
        self._add_knowledge_node("transfer_learning", "Transfer Learning", "computer_vision", "advanced")
        self._add_knowledge_node("object_detection", "Object Detection", "computer_vision", "advanced")
        self._add_knowledge_node("segmentation", "Image Segmentation", "computer_vision", "advanced")

        # NLP
        self._add_knowledge_node("text_processing", "Text Processing", "nlp", "beginner")
        self._add_knowledge_node("word_embeddings", "Word Embeddings", "nlp", "intermediate")
        self._add_knowledge_node("rnns", "Recurrent Neural Networks", "nlp", "intermediate")
        self._add_knowledge_node("lstm_gru", "LSTM & GRU", "nlp", "intermediate")
        self._add_knowledge_node("attention", "Attention Mechanism", "nlp", "advanced")
        self._add_knowledge_node("transformers", "Transformers", "nlp", "advanced")
        self._add_knowledge_node("bert", "BERT", "nlp", "advanced")
        self._add_knowledge_node("gpt", "GPT Models", "nlp", "advanced")

        # Reinforcement Learning
        self._add_knowledge_node("rl_intro", "RL Introduction", "reinforcement_learning", "intermediate")
        self._add_knowledge_node("mdp", "Markov Decision Processes", "reinforcement_learning", "intermediate")
        self._add_knowledge_node("q_learning", "Q-Learning", "reinforcement_learning", "advanced")
        self._add_knowledge_node("policy_gradients", "Policy Gradients", "reinforcement_learning", "advanced")
        self._add_knowledge_node("dqn", "Deep Q-Networks", "reinforcement_learning", "advanced")

        # MLOps
        self._add_knowledge_node("model_deployment", "Model Deployment", "mlops", "advanced")
        self._add_knowledge_node("model_monitoring", "Model Monitoring", "mlops", "advanced")
        self._add_knowledge_node("mlflow", "MLflow", "mlops", "advanced")
        self._add_knowledge_node("docker_ml", "Docker for ML", "mlops", "advanced")

        # Define prerequisites (directed edges)
        prerequisites = [
            # Python basics are fundamental
            ("python_basics", "numpy"),
            ("python_basics", "pandas"),
            ("python_basics", "matplotlib"),

            # Math prerequisites for ML
            ("linear_algebra", "ml_intro"),
            ("statistics", "ml_intro"),
            ("python_basics", "ml_intro"),

            # ML prerequisites
            ("ml_intro", "supervised_learning"),
            ("ml_intro", "unsupervised_learning"),
            ("supervised_learning", "linear_regression"),
            ("supervised_learning", "logistic_regression"),
            ("supervised_learning", "decision_trees"),
            ("decision_trees", "random_forests"),
            ("supervised_learning", "svm"),
            ("unsupervised_learning", "clustering"),
            ("unsupervised_learning", "dimensionality_reduction"),

            # Deep Learning prerequisites
            ("linear_algebra", "neural_networks"),
            ("calculus", "backpropagation"),
            ("supervised_learning", "neural_networks"),
            ("neural_networks", "activation_functions"),
            ("neural_networks", "backpropagation"),
            ("backpropagation", "optimization"),
            ("neural_networks", "regularization"),
            ("regularization", "dropout"),
            ("regularization", "batch_norm"),

            # Computer Vision prerequisites
            ("numpy", "image_processing"),
            ("matplotlib", "image_processing"),
            ("neural_networks", "cnns"),
            ("cnns", "conv_layers"),
            ("cnns", "pooling"),
            ("cnns", "transfer_learning"),
            ("transfer_learning", "object_detection"),
            ("cnns", "segmentation"),

            # NLP prerequisites
            ("python_basics", "text_processing"),
            ("neural_networks", "word_embeddings"),
            ("neural_networks", "rnns"),
            ("rnns", "lstm_gru"),
            ("lstm_gru", "attention"),
            ("attention", "transformers"),
            ("transformers", "bert"),
            ("transformers", "gpt"),

            # RL prerequisites
            ("probability", "rl_intro"),
            ("rl_intro", "mdp"),
            ("mdp", "q_learning"),
            ("mdp", "policy_gradients"),
            ("q_learning", "dqn"),
            ("neural_networks", "dqn"),

            # MLOps prerequisites
            ("supervised_learning", "model_deployment"),
            ("model_deployment", "model_monitoring"),
            ("model_deployment", "mlflow"),
            ("model_deployment", "docker_ml"),
        ]

        for prereq, dependent in prerequisites:
            self.add_prerequisite(prereq, dependent)

    def _add_knowledge_node(self, node_id: str, title: str, category: str, difficulty: str):
        """Add a knowledge node to the graph"""
        node = KnowledgeNode(node_id, title, category, difficulty)
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **{
            "title": title,
            "category": category,
            "difficulty": difficulty
        })

        # Unlock beginner nodes by default
        if difficulty == "beginner":
            node.unlocked = True

    def add_prerequisite(self, prerequisite_id: str, dependent_id: str):
        """Add a prerequisite relationship"""
        if prerequisite_id in self.nodes and dependent_id in self.nodes:
            self.nodes[dependent_id].prerequisites.add(prerequisite_id)
            self.nodes[prerequisite_id].dependents.add(dependent_id)
            self.graph.add_edge(prerequisite_id, dependent_id)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge node by ID"""
        return self.nodes.get(node_id)

    def update_mastery(self, node_id: str, mastery_score: float):
        """Update mastery score for a node"""
        node = self.get_node(node_id)
        if node:
            node.update_mastery(mastery_score)

            # Unlock dependent nodes if mastery threshold is met
            if mastery_score >= 0.7:
                for dependent_id in node.dependents:
                    dependent = self.get_node(dependent_id)
                    if dependent and dependent.is_ready_to_learn(self):
                        dependent.unlocked = True

    def get_next_topics(self, mastered_nodes: Set[str], limit: int = 5) -> List[KnowledgeNode]:
        """Get next recommended topics based on current progress"""
        recommendations = []

        for node_id, node in self.nodes.items():
            if node_id in mastered_nodes:
                continue

            if node.is_ready_to_learn(self):
                recommendations.append(node)

        # Sort by difficulty and category
        difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        recommendations.sort(key=lambda n: (
            difficulty_order.get(n.difficulty, 999),
            -len(n.prerequisites)  # Prefer topics with more prerequisites satisfied
        ))

        return recommendations[:limit]

    def get_learning_path_to(self, target_node_id: str, current_mastery: Set[str]) -> List[str]:
        """Get shortest learning path to a target node"""
        target = self.get_node(target_node_id)
        if not target:
            return []

        # Find all prerequisite paths
        all_paths = []
        for prereq_id in target.prerequisites:
            if prereq_id not in current_mastery:
                sub_path = self.get_learning_path_to(prereq_id, current_mastery)
                all_paths.extend(sub_path)

        # Add target if prerequisites are met or will be met by this path
        all_paths.append(target_node_id)

        # Remove duplicates while preserving order
        seen = set()
        ordered_path = []
        for node_id in all_paths:
            if node_id not in seen:
                seen.add(node_id)
                ordered_path.append(node_id)

        return ordered_path

    def get_category_progress(self, category: str, user_mastery: Dict[str, float]) -> Dict[str, Any]:
        """Get progress statistics for a category"""
        category_nodes = [n for n in self.nodes.values() if n.category == category]

        if not category_nodes:
            return {"total": 0, "mastered": 0, "in_progress": 0, "locked": 0}

        total = len(category_nodes)
        mastered = sum(1 for n in category_nodes if user_mastery.get(n.node_id, 0) >= 0.8)
        in_progress = sum(1 for n in category_nodes
                         if 0.3 <= user_mastery.get(n.node_id, 0) < 0.8)
        locked = sum(1 for n in category_nodes if not n.unlocked)

        return {
            "category": category,
            "total": total,
            "mastered": mastered,
            "in_progress": in_progress,
            "locked": locked,
            "completion_rate": mastered / total if total > 0 else 0
        }

    def get_graph_visualization_data(self, user_mastery: Dict[str, float]) -> Dict[str, Any]:
        """Get data for visualizing the knowledge graph"""
        nodes = []
        edges = []

        for node_id, node in self.nodes.items():
            mastery = user_mastery.get(node_id, 0.0)

            nodes.append({
                "id": node_id,
                "label": node.title,
                "category": node.category,
                "difficulty": node.difficulty,
                "mastery": mastery,
                "unlocked": node.unlocked,
                "status": "mastered" if mastery >= 0.8 else "in_progress" if mastery >= 0.3 else "locked"
            })

        for source, target in self.graph.edges():
            edges.append({
                "source": source,
                "target": target
            })

        return {
            "nodes": nodes,
            "edges": edges
        }

    def get_knowledge_stats(self, user_mastery: Dict[str, float]) -> Dict[str, Any]:
        """Get overall knowledge statistics"""
        total_nodes = len(self.nodes)
        mastered = sum(1 for score in user_mastery.values() if score >= 0.8)
        in_progress = sum(1 for score in user_mastery.values() if 0.3 <= score < 0.8)

        categories = set(n.category for n in self.nodes.values())
        category_progress = {
            cat: self.get_category_progress(cat, user_mastery)
            for cat in categories
        }

        return {
            "total_topics": total_nodes,
            "mastered_topics": mastered,
            "in_progress_topics": in_progress,
            "not_started": total_nodes - mastered - in_progress,
            "overall_progress": mastered / total_nodes if total_nodes > 0 else 0,
            "categories": category_progress
        }
