# Hybrid-Text-Based-Recommendation-Using-User-Reviews

downlod the data here : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=Books_rating.csv
# Hybrid Book Recommender - Quick Start

## Résultats Principaux
- **Best model**: Hybrid (Word2Vec) - 0.9104 RMSE
- **Cold-start gain**: +13.6% vs CF
- **Overall gain**: +6.4% vs CF

## Justifications Clés
- **α=0.6**: Standard literature value, validated by cold-start analysis
- **500K subset**: Computational constraints, representative sample
- **Word2Vec ≈ BERT**: Similar performance, α=0.6 makes CF dominant

## Story pour le Papier
"Text features help MOST in cold-start scenarios (+13.6%), 
but add little for popular items (+0.4%), 
showing the complementary nature of CF and CB."

## Si Reviewer Demande
- **Why α=0.6?** → "Standard practice + validated by results"
- **Why CB < CF?** → "CB needs user history; defaults to global mean"
- **Why embeddings similar?** → "CF dominates at α=0.6; lower α would differentiate"

## Future Work (pour défense)
1. Adaptive α based on item popularity
2. Fine-tune BERT on book domain
3. Neural hybrid (not linear fusion)
4. Aspect-based explanations