from src.recommender import load_ratings, build_clustered_collab_model
from src.evaluate import split_user_ratings

ratings      = load_ratings(sample=True)
train, test  = split_user_ratings(ratings)
cluster_data = build_clustered_collab_model(train)
user_cluster = cluster_data['user_cluster']
test_users   = test['userId'].unique()
in_cluster   = sum(1 for u in test_users if u in user_cluster)

print(f'Test users total     : {len(test_users)}')
print(f'Test users in cluster: {in_cluster}')
print(f'Coverage             : {in_cluster/len(test_users):.1%}')

liked = test[test['rating'] >= 4.0]
print(f'Test ratings >= 4.0  : {len(liked)}')
print(f'Avg liked per user   : {liked.groupby("userId").size().mean():.1f}')