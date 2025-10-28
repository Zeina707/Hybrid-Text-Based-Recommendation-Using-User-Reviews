import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import gc

print("="*70)
print("PHASE 3: HYBRID RECOMMENDATION - 1M REVIEW SUBSET")
print("="*70)

# ============================================================
# STEP 1: LOAD FULL DATA FROM PHASE 2, THEN SAMPLE TO 1M
# ============================================================
print("\n[1/7] Loading data and sampling to 1M reviews...")

df_train_full = pd.read_csv('train_data.csv')
print(f"  Full dataset: {len(df_train_full):,} reviews")

# Sample 1M reviews
sample_size = min(1_000_000, len(df_train_full))
df_train = df_train_full.sample(n=sample_size, random_state=42)
print(f"✓ Sampled: {len(df_train):,} reviews")
print(f"  Items: {df_train['item_id'].nunique():,}")
print(f"  Users: {df_train['user_id'].nunique():,}")

# Load all item features (we'll filter below)
item_features_full = pd.read_csv('item_features.csv')
print(f"✓ Item features: {len(item_features_full):,} items")

# Keep only items that appear in sampled data
items_in_sample = set(df_train['item_id'].unique())
item_features = item_features_full[item_features_full['item_id'].isin(items_in_sample)].reset_index(drop=True)
print(f"✓ Filtered to: {len(item_features):,} items in sample")

# Load embeddings (only for items in sample)
doc_embeddings_full = np.load('doc_embeddings.npy')
# We need to map which rows correspond to items in our sample
# Load the mapping from Phase 2
with open('w2v_model.pkl', 'rb') as f:
    w2v_model = pickle.load(f)

print(f"✓ Loaded embeddings: {doc_embeddings_full.shape}")

# ============================================================
# STEP 2: BUILD COLLABORATIVE FILTERING
# ============================================================
print("\n[2/7] Training Collaborative Filtering (SVD)...")

user_to_idx = {uid: i for i, uid in enumerate(df_train['user_id'].unique())}
item_to_idx = {iid: i for i, iid in enumerate(df_train['item_id'].unique())}

print(f"  User-Item Matrix: ({len(user_to_idx):,}, {len(item_to_idx):,})")

row_indices = df_train['user_id'].map(user_to_idx).values
col_indices = df_train['item_id'].map(item_to_idx).values
ratings = df_train['rating'].values

user_item_matrix = csr_matrix(
    (ratings, (row_indices, col_indices)),
    shape=(len(user_to_idx), len(item_to_idx))
)

print(f"  Sparsity: {(1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%")

global_mean = ratings.mean()
print(f"  Global mean rating: {global_mean:.2f}")

print("  Computing biases...")
user_bias = (df_train.groupby('user_id')['rating'].mean() - global_mean).to_dict()
item_bias = (df_train.groupby('item_id')['rating'].mean() - global_mean).to_dict()

for uid in user_to_idx.keys():
    if uid not in user_bias:
        user_bias[uid] = 0.0
for iid in item_to_idx.keys():
    if iid not in item_bias:
        item_bias[iid] = 0.0

print("  Training SVD (n_components=50)...")
n_factors = 50
svd_model = TruncatedSVD(n_components=n_factors, random_state=42, n_iter=10)
user_factors = svd_model.fit_transform(user_item_matrix)
item_factors = svd_model.components_.T

print(f"✓ SVD trained")
print(f"  Explained variance: {svd_model.explained_variance_ratio_.sum():.4f}")
print(f"  User factors: {user_factors.shape}")
print(f"  Item factors: {item_factors.shape}")

# ============================================================
# STEP 3: BUILD CONTENT-BASED FEATURES
# ============================================================
print("\n[3/7] Building content-based features...")

scaler = StandardScaler()

# Sentiment
sentiment_features = item_features[['sentiment_polarity', 'sentiment_subjectivity']].fillna(0).values
sentiment_norm = scaler.fit_transform(sentiment_features)

# Topics
topic_features = item_features[[f'topic_{i}' for i in range(15)]].fillna(0).values

# Aspects
aspect_cols = ['plot', 'characters', 'writing_style', 'setting', 'emotion', 'pacing']
aspect_features = item_features[[f'aspect_{asp}' for asp in aspect_cols]].fillna(0).values

# Embeddings (100 dims)
embedding_cols = [f'embedding_{i}' for i in range(100)]
embedding_features = item_features[embedding_cols].fillna(0).astype(np.float32).values

# Combine weighted
combined_content = np.hstack([
    sentiment_norm * 0.1,
    topic_features * 0.2,
    aspect_features * 0.3,
    embedding_features * 0.4
]).astype(np.float32)

content_scaler = StandardScaler()
content_features_norm = content_scaler.fit_transform(combined_content)
content_features_norm = normalize(content_features_norm, norm='l2').astype(np.float32)

print(f"✓ Content features: {content_features_norm.shape}")
print(f"  Memory: ~{content_features_norm.nbytes / (1024**2):.1f} MB")

del sentiment_features, topic_features, aspect_features, embedding_features, combined_content
gc.collect()

# ============================================================
# STEP 4: COMPUTE ITEM SIMILARITY (NOW MANAGEABLE!)
# ============================================================
print("\n[4/7] Computing item similarity...")

batch_size = 2000
n_items = len(item_features)
item_similarity_dict = {}

print(f"  Processing {n_items:,} items in batches of {batch_size}...")

for i in range(0, n_items, batch_size):
    batch_end = min(i + batch_size, n_items)
    batch_features = content_features_norm[i:batch_end]
    
    # Compute similarity with all items
    batch_similarity = cosine_similarity(batch_features, content_features_norm)
    
    # Store ALL similarities (threshold 0.1)
    for local_idx in range(batch_similarity.shape[0]):
        global_idx = i + local_idx
        sims = batch_similarity[local_idx]
        
        min_sim_threshold = 0.1
        valid_mask = sims > min_sim_threshold
        valid_indices = np.where(valid_mask)[0]
        valid_sims = sims[valid_indices]
        
        if len(valid_sims) > 0:
            sort_order = np.argsort(valid_sims)[::-1]
            
            item_similarity_dict[global_idx] = {
                'indices': valid_indices[sort_order].astype(np.int32),
                'similarities': valid_sims[sort_order].astype(np.float32)
            }
    
    if (i + batch_size) % 5000 == 0 or batch_end == n_items:
        print(f"  ✓ Processed {batch_end:,}/{n_items:,} items")

print(f"✓ Similarity index created")
avg_sims = np.mean([len(v['indices']) for v in item_similarity_dict.values()])
total_mem = (sum(len(v['indices']) for v in item_similarity_dict.values()) * 8) / (1024**2)
print(f"  Avg similarities per item: {avg_sims:.0f}")
print(f"  Total memory: ~{total_mem:.1f} MB")

del content_features_norm
gc.collect()

# ============================================================
# STEP 5: HYBRID RECOMMENDER CLASS
# ============================================================
print("\n[5/7] Building hybrid recommender...")

class HybridRecommender:
    def __init__(self, user_factors, item_factors, user_to_idx, item_to_idx,
                 item_similarity_dict, item_features, df_train,
                 user_bias, item_bias, global_mean, alpha=0.5):
        
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.item_similarity_dict = item_similarity_dict
        self.item_features = item_features
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_mean = global_mean
        self.alpha = alpha
        
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}
        self.user_items = df_train.groupby('user_id')['item_id'].apply(set).to_dict()
        
        self.user_item_ratings = {}
        for uid in self.user_to_idx.keys():
            self.user_item_ratings[uid] = {}
        
        for _, row in df_train.iterrows():
            uid = row['user_id']
            iid = row['item_id']
            if iid in item_to_idx:
                self.user_item_ratings[uid][self.item_to_idx[iid]] = row['rating']
    
    def get_cf_score(self, user_id, item_id):
        """CF score with bias adjustment"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[item_id]
        
        dot_product = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        prediction = (self.global_mean + 
                     self.user_bias.get(user_id, 0.0) + 
                     self.item_bias.get(item_id, 0.0) + 
                     dot_product)
        
        return np.clip(prediction, 1.0, 5.0)
    
    def get_content_score(self, user_id, item_id):
        """Content-based score using similarities"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        item_idx = self.item_to_idx[item_id]
        user_rated_items = self.user_items.get(user_id, set())
        
        if not user_rated_items or item_idx not in self.item_similarity_dict:
            return self.global_mean
        
        user_item_indices = {self.item_to_idx[iid]: iid 
                            for iid in user_rated_items 
                            if iid in self.item_to_idx}
        
        if not user_item_indices:
            return self.global_mean
        
        similar_data = self.item_similarity_dict[item_idx]
        similar_indices = similar_data['indices']
        similar_sims = similar_data['similarities'].astype(np.float32)
        
        weighted_sum = 0.0
        sim_sum = 0.0
        
        for sim_idx, sim_val in zip(similar_indices, similar_sims):
            if sim_idx in user_item_indices:
                rating = self.user_item_ratings[user_id].get(sim_idx, self.global_mean)
                weighted_sum += float(sim_val) * rating
                sim_sum += float(sim_val)
        
        if sim_sum == 0:
            return self.global_mean
        
        prediction = weighted_sum / sim_sum
        return np.clip(prediction, 1.0, 5.0)

recommender = HybridRecommender(
    user_factors, item_factors, user_to_idx, item_to_idx,
    item_similarity_dict, item_features, df_train,
    user_bias, item_bias, global_mean, alpha=0.5
)
print("✓ Recommender initialized (α=0.5)")

# ============================================================
# STEP 6: EVALUATION
# ============================================================
print("\n[6/7] Evaluating performance...")

sample_size = min(10000, len(df_train))
test_sample = df_train.sample(n=sample_size, random_state=42)

cf_preds = []
content_preds = []
hybrid_preds = []
actual_ratings = []

print(f"Scoring {sample_size:,} predictions...")
for idx, (_, row) in enumerate(test_sample.iterrows()):
    if idx % 2000 == 0:
        print(f"  ✓ {idx:,}/{sample_size:,}")
    
    cf = recommender.get_cf_score(row['user_id'], row['item_id'])
    content = recommender.get_content_score(row['user_id'], row['item_id'])
    hybrid = 0.5 * cf + 0.5 * content
    
    cf_preds.append(cf)
    content_preds.append(content)
    hybrid_preds.append(hybrid)
    actual_ratings.append(row['rating'])

cf_rmse = np.sqrt(mean_squared_error(actual_ratings, cf_preds))
content_rmse = np.sqrt(mean_squared_error(actual_ratings, content_preds))
hybrid_rmse = np.sqrt(mean_squared_error(actual_ratings, hybrid_preds))

print(f"\n✓ Performance Metrics ({sample_size:,} samples):")
print(f"  CF Only:      RMSE = {cf_rmse:.4f}")
print(f"  Content Only: RMSE = {content_rmse:.4f}")
print(f"  Hybrid (50%): RMSE = {hybrid_rmse:.4f}")

print(f"\n  Score distributions:")
print(f"    CF:       mean={np.mean(cf_preds):.2f}, std={np.std(cf_preds):.2f}")
print(f"    Content:  mean={np.mean(content_preds):.2f}, std={np.std(content_preds):.2f}")
print(f"    Hybrid:   mean={np.mean(hybrid_preds):.2f}, std={np.std(hybrid_preds):.2f}")
print(f"    Actual:   mean={np.mean(actual_ratings):.2f}, std={np.std(actual_ratings):.2f}")

# ============================================================
# STEP 7: SAVE MODELS
# ============================================================
print("\n[7/7] Saving models...")

joblib.dump(user_factors, 'user_factors_1m.pkl', compress=3)
joblib.dump(item_factors, 'item_factors_1m.pkl', compress=3)
joblib.dump(user_to_idx, 'user_to_idx_1m.pkl', compress=3)
joblib.dump(item_to_idx, 'item_to_idx_1m.pkl', compress=3)
joblib.dump(user_bias, 'user_bias_1m.pkl', compress=3)
joblib.dump(item_bias, 'item_bias_1m.pkl', compress=3)
joblib.dump(recommender, 'hybrid_recommender_1m.pkl', compress=3)
joblib.dump(item_similarity_dict, 'item_similarity_dict_1m.pkl', compress=3)

print("✓ All models saved (with _1m suffix)")

print("\n" + "="*70)
print("PHASE 3 COMPLETE!")
print("="*70)
print(f"\n✅ Used 1M review subset (from full {len(df_train_full):,})")
print(f"   Phases 1 & 2 remain unchanged")
print(f"   Phase 3 now runs without memory issues!")
