from datetime import datetime
import os
from typing import Dict, List, Optional, Union
import uuid
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()

class EmbeddingProvider:
    """Abstract base class for embedding providers"""
    def get_text_embedding(self, text: str) -> np.ndarray:
        raise NotImplementedError

class NVIDIAEmbeddingProvider(EmbeddingProvider):
    """NVIDIA NIM embedding provider"""
    def __init__(self, model_name: str, nim_endpoint: str, api_key: str):
        from llama_index.embeddings.nvidia import NVIDIAEmbedding
        self.embed_model = NVIDIAEmbedding(
            model_name=model_name,
            nim_endpoint=nim_endpoint,
            api_key=api_key
        )
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        return self.embed_model.get_text_embedding(text)

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local HuggingFace model for embeddings"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        # Tokenize and move to device
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Move to CPU and convert to numpy
        return embeddings.cpu().numpy().flatten()

class EnhancedKnowledgeBase:
    def __init__(self, nim_endpoint: str = None):
        """Initialize knowledge base with fallback embedding options"""
        self.patterns = {}
        self.pattern_relationships = {}
        self.pattern_clusters = {}
        self.embeddings = {}
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.feature_buffer = []
        
        # Try to initialize embedding providers in order of preference
        self.embedding_provider = None
        
        if nim_endpoint:
            try:
                self.embedding_provider = NVIDIAEmbeddingProvider(
                    model_name="nvolveqa_40k",
                    nim_endpoint=nim_endpoint,
                    api_key=os.environ.get("NVIDIA_API_KEY", "")
                )
                print("Successfully initialized NVIDIA NIM embeddings")
            except Exception as e:
                print(f"Warning: Failed to initialize NVIDIA NIM embeddings: {str(e)}")
        
        if self.embedding_provider is None:
            try:
                self.embedding_provider = LocalEmbeddingProvider()
                print("Successfully initialized local embedding model")
            except Exception as e:
                print(f"Warning: Failed to initialize local embeddings: {str(e)}")
                print("Falling back to feature-based similarity")
        
    def add_pattern(self, pattern: Dict) -> str:
        """Add pattern with embedding if available"""
        pattern_id = str(uuid.uuid4())
        
        try:
            # Standardize pattern storage
            stored_pattern = {
                'pattern': {
                    'features': self._convert_to_numpy(pattern['features']),
                    'predictions': self._convert_to_numpy(pattern['predictions']),
                    'uncertainty': pattern['uncertainty'],
                    'timestamp': pattern.get('timestamp', datetime.now())
                },
                'timestamp': datetime.now(),
                'id': pattern_id
            }
            
            # Try to get embedding if provider is available
            if self.embedding_provider:
                try:
                    text_representation = self._pattern_to_text(pattern)
                    embedding = self.embedding_provider.get_text_embedding(text_representation)
                    self.embeddings[pattern_id] = embedding
                except Exception as e:
                    print(f"Warning: Failed to get embedding: {str(e)}")
            
            # Store pattern
            self.patterns[pattern_id] = stored_pattern
            
            # Update feature buffer for clustering
            self.feature_buffer.append(self._flatten_features(pattern['features']))
            if len(self.feature_buffer) > 100:
                self.feature_buffer = self.feature_buffer[-100:]
            
            # Update clustering if enough data
            if len(self.feature_buffer) >= 5:
                self._update_clusters()
                
        except Exception as e:
            print(f"Warning: Error in pattern storage: {str(e)}")
            # Ensure basic storage still works
            self.patterns[pattern_id] = {
                'pattern': pattern,
                'timestamp': datetime.now(),
                'id': pattern_id
            }
        
        return pattern_id
    
    def find_similar_patterns(self, pattern: Dict, threshold: float = 0.8) -> List[Dict]:
        """Find similar patterns using available embedding method"""
        similar_patterns = []
        
        try:
            if self.embedding_provider and self.embeddings:
                # Try using embeddings first
                try:
                    query_text = self._pattern_to_text(pattern)
                    query_embedding = self.embedding_provider.get_text_embedding(query_text)
                    
                    # Compare with stored embeddings
                    for stored_id, stored_pattern in self.patterns.items():
                        if stored_id in self.embeddings:
                            similarity = self._cosine_similarity(
                                query_embedding,
                                self.embeddings[stored_id]
                            )
                            
                            if similarity > threshold:
                                similar_patterns.append({
                                    'pattern': stored_pattern['pattern'],
                                    'similarity': float(similarity)
                                })
                    
                    if similar_patterns:  # If we found patterns using embeddings
                        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
                        return similar_patterns[:5]
                        
                except Exception as e:
                    print(f"Warning: Embedding similarity search failed: {str(e)}")
            
            # Fallback to feature-based comparison
            return self._basic_similarity_search(pattern, threshold)
            
        except Exception as e:
            print(f"Warning: Error in similarity search: {str(e)}")
            return []

    def analyze_pattern(self, pattern_id: str, similar_patterns: List[Dict]) -> Dict:
        """Analyze pattern with comprehensive metrics"""
        try:
            # Calculate trend
            trend = self._analyze_trend(similar_patterns)
            
            # Get cluster assignment
            cluster = self._get_cluster_assignment(pattern_id)
            
            # Calculate anomaly scores
            anomaly_score = self._calculate_anomaly_score(similar_patterns)
            
            # Add embedding-based analysis if available
            if self.embedding_provider and pattern_id in self.embeddings:
                try:
                    embedding_analysis = self._analyze_embedding(pattern_id)
                    trend['embedding_confidence'] = embedding_analysis.get('confidence', 0.0)
                    anomaly_score = (anomaly_score + embedding_analysis.get('anomaly_score', 0.0)) / 2
                except Exception as e:
                    print(f"Warning: Embedding analysis failed: {str(e)}")
            
            return {
                'trend': trend,
                'cluster': cluster,
                'anomaly_score': anomaly_score
            }
            
        except Exception as e:
            print(f"Warning: Error in pattern analysis: {str(e)}")
            return {
                'trend': {'type': 'unknown', 'confidence': 0.0},
                'cluster': {'id': 'unknown', 'similarity': 0.0},
                'anomaly_score': 0.5
            }

    def _analyze_embedding(self, pattern_id: str) -> Dict:
        """Analyze pattern using embeddings"""
        if not self.embedding_provider or pattern_id not in self.embeddings:
            return {'confidence': 0.0, 'anomaly_score': 0.5}
        
        try:
            embedding = self.embeddings[pattern_id]
            
            # Calculate embedding statistics
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            
            # Calculate confidence based on embedding properties
            confidence = min(1.0, embedding_norm / np.sqrt(len(embedding)))
            
            # Calculate anomaly score based on deviation from typical embeddings
            all_means = [np.mean(emb) for emb in self.embeddings.values()]
            all_stds = [np.std(emb) for emb in self.embeddings.values()]
            
            mean_diff = abs(embedding_mean - np.mean(all_means))
            std_diff = abs(embedding_std - np.mean(all_stds))
            
            anomaly_score = (mean_diff + std_diff) / 2
            
            return {
                'confidence': float(confidence),
                'anomaly_score': float(min(1.0, anomaly_score))
            }
            
        except Exception as e:
            print(f"Warning: Error in embedding analysis: {str(e)}")
            return {'confidence': 0.0, 'anomaly_score': 0.5}

    # [Previous helper methods remain unchanged]
    def _pattern_to_text(self, pattern: Dict) -> str:
        """Convert pattern to text representation"""
        features = pattern['features'].tolist() if isinstance(pattern['features'], np.ndarray) else pattern['features']
        predictions = pattern['predictions'].tolist() if isinstance(pattern['predictions'], np.ndarray) else pattern['predictions']
        
        return f"""
        Pattern Analysis:
        Features: {features}
        Predictions: {predictions}
        Uncertainty Values: {pattern['uncertainty']}
        Timestamp: {pattern.get('timestamp', datetime.now())}
        """

    def _basic_similarity_search(self, pattern: Dict, threshold: float) -> List[Dict]:
        """Basic similarity search using euclidean distance"""
        similar_patterns = []
        pattern_features = np.array(pattern['features']).flatten()
        
        for stored_pattern in self.patterns.values():
            stored_features = np.array(stored_pattern['pattern']['features']).flatten()
            
            # Calculate similarity using euclidean distance
            distance = np.linalg.norm(pattern_features - stored_features)
            similarity = 1 / (1 + distance)  # Convert distance to similarity score
            
            if similarity > threshold:
                similar_patterns.append({
                    'pattern': stored_pattern['pattern'],
                    'similarity': float(similarity)
                })
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_patterns[:5]  # Return top 5 similar patterns

    def _convert_to_numpy(self, data) -> np.ndarray:
        """Safely convert data to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array([data])

    def _flatten_features(self, features) -> np.ndarray:
        """Safely flatten feature array"""
        features_array = self._convert_to_numpy(features)
        return features_array.flatten()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity with error handling"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            print(f"Warning: Error calculating similarity: {str(e)}")
            return 0.0
        
# [Previous code remains the same until class EnhancedKnowledgeBase's last method]

    def _analyze_trend(self, similar_patterns: List[Dict]) -> Dict:
        """Analyze trend with robust error handling"""
        if not similar_patterns:
            return {'type': 'new_pattern', 'confidence': 1.0}
            
        try:
            similarities = [p['similarity'] for p in similar_patterns]
            
            if len(similarities) >= 3:
                trend = np.mean(similarities[-3:]) - np.mean(similarities[:3])
                
                if trend > 0.1:
                    trend_type = 'improving'
                elif trend < -0.1:
                    trend_type = 'degrading'
                else:
                    trend_type = 'stable'
                    
                confidence = float(np.mean(similarities))
            else:
                trend_type = 'insufficient_data'
                confidence = 0.5
                
            return {
                'type': trend_type,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Warning: Error in trend analysis: {str(e)}")
            return {'type': 'unknown', 'confidence': 0.0}

    def _update_clusters(self):
        """Update cluster assignments"""
        try:
            if len(self.feature_buffer) < 5:  # Minimum samples for clustering
                return
                
            X = np.array(self.feature_buffer)
            X_scaled = self.scaler.fit_transform(X)
            
            # Update number of clusters based on data size
            n_clusters = min(5, len(X) // 2)
            if n_clusters != self.kmeans.n_clusters:
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            
            cluster_labels = self.kmeans.fit_predict(X_scaled)
            
            # Update cluster assignments for recent patterns
            pattern_ids = list(self.patterns.keys())[-len(self.feature_buffer):]
            for pattern_id, cluster_id in zip(pattern_ids, cluster_labels):
                self.pattern_clusters[pattern_id] = int(cluster_id)
                
        except Exception as e:
            print(f"Warning: Error updating clusters: {str(e)}")

    def _get_cluster_assignment(self, pattern_id: str) -> Dict:
        """Get cluster assignment with fallback"""
        try:
            if pattern_id in self.pattern_clusters:
                cluster_id = self.pattern_clusters[pattern_id]
                return {
                    'id': f'cluster_{cluster_id}',
                    'size': len([pid for pid, cid in self.pattern_clusters.items() 
                               if cid == cluster_id]),
                    'similarity': 1.0
                }
            return {'id': 'unknown', 'size': 0, 'similarity': 0.0}
        except Exception as e:
            print(f"Warning: Error in cluster assignment: {str(e)}")
            return {'id': 'unknown', 'size': 0, 'similarity': 0.0}

    def _calculate_anomaly_score(self, similar_patterns: List[Dict]) -> float:
        """Calculate anomaly score based on pattern similarities"""
        try:
            if not similar_patterns:
                return 1.0  # Completely new pattern is considered anomalous
            
            similarities = [p['similarity'] for p in similar_patterns]
            mean_similarity = np.mean(similarities)
            
            # Calculate variance in similarities for uncertainty
            similarity_std = np.std(similarities) if len(similarities) > 1 else 0
            
            # Combine mean dissimilarity and uncertainty
            anomaly_score = (1.0 - mean_similarity) * (1 + similarity_std)
            
            return float(min(1.0, anomaly_score))
            
        except Exception as e:
            print(f"Warning: Error calculating anomaly score: {str(e)}")
            return 0.5

    def get_cluster_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each cluster"""
        try:
            cluster_stats = {}
            
            for cluster_id in set(self.pattern_clusters.values()):
                # Get patterns in this cluster
                cluster_patterns = [
                    pid for pid, cid in self.pattern_clusters.items()
                    if cid == cluster_id
                ]
                
                # Calculate cluster statistics
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_patterns),
                    'average_similarity': self._calculate_cluster_cohesion(cluster_patterns),
                    'latest_pattern': self._get_latest_pattern_in_cluster(cluster_patterns),
                    'creation_time': self._get_cluster_creation_time(cluster_patterns)
                }
            
            return cluster_stats
            
        except Exception as e:
            print(f"Warning: Error calculating cluster statistics: {str(e)}")
            return {}

    def _calculate_cluster_cohesion(self, pattern_ids: List[str]) -> float:
        """Calculate average similarity between patterns in a cluster"""
        try:
            if len(pattern_ids) < 2:
                return 1.0
                
            similarities = []
            for i, pid1 in enumerate(pattern_ids[:-1]):
                for pid2 in pattern_ids[i+1:]:
                    if pid1 in self.embeddings and pid2 in self.embeddings:
                        similarity = self._cosine_similarity(
                            self.embeddings[pid1],
                            self.embeddings[pid2]
                        )
                        similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            print(f"Warning: Error calculating cluster cohesion: {str(e)}")
            return 0.0

    def _get_latest_pattern_in_cluster(self, pattern_ids: List[str]) -> Optional[str]:
        """Get the most recent pattern in a cluster"""
        try:
            if not pattern_ids:
                return None
                
            latest_time = datetime.min
            latest_pattern = None
            
            for pid in pattern_ids:
                if pid in self.patterns:
                    pattern_time = self.patterns[pid]['timestamp']
                    if pattern_time > latest_time:
                        latest_time = pattern_time
                        latest_pattern = pid
            
            return latest_pattern
            
        except Exception as e:
            print(f"Warning: Error finding latest pattern: {str(e)}")
            return None

    def _get_cluster_creation_time(self, pattern_ids: List[str]) -> datetime:
        """Get the creation time of the cluster"""
        try:
            if not pattern_ids:
                return datetime.now()
                
            creation_times = [
                self.patterns[pid]['timestamp']
                for pid in pattern_ids
                if pid in self.patterns
            ]
            
            return min(creation_times) if creation_times else datetime.now()
            
        except Exception as e:
            print(f"Warning: Error getting cluster creation time: {str(e)}")
            return datetime.now()        