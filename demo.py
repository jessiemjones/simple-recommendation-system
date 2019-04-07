import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it.
data = fetch_movielens(min_rating=4.0)

#printtraining and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
#warp = weighted approximate-rank pairwise
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendtaion(model, data, user_ids):
    #first get num of users and movies in training data
    n_users, n_items = data['train'].shape

    for user_id in user_ids:

        #movies they arlready like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts the user will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print results
        print('User %s' % user_id)
        print('       Known postitives:')

        for x in known_positives[:3]:
            print('             %s' % x)

        print('       Recommended:')

        for x in top_items[:3]:
            print('             %s' % x)

sample_recommendtaion(model, data, [3, 25, 450])