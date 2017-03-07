    model                                                           |   test    |   train        |   vectorizer |   dataset |
--------------------------------------------------------------------|-----------|----------------|--------------|-----------|
ExtraTreesClassifier 100 max_leaf_nodes=2500, min_samples_leaf=5    |   0.76    |   0.81         |   5000 tfidf |   rotten  |
ExtraTreesClassifier 50                                             |   0.737   |  0.985         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 100                                            |   0.739   |  0.985         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 50                                             |   0.719   |  0.958         |   1000 count |   rotten  |
ExtraTreesClassifier 200                                            |   0.739   |  0.985         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 100 min_samples_leaf=50                        |   0.715   |  0.719         |   1000 tfidf |   rotten  |
AdaBoostClassifier 700 learning_rate=0.2                            |   0.712   |  0.714         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 100 min_samples_leaf=20                        |   0.727   |  0.743         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 100 min_samples_leaf=10                        |   0.735   |  0.766         |   1000 tfidf |   rotten  |
ExtraTreesClassifier 100                                            |   0.775   |  0.999         |   20000 tfidf|   rotten  |
AdaBoostClassifier 1500                                             |   0.764   |  0.802         |   20000 tfidf|   rotten  |
AdaBoostClassifier 1500 learning_rate=0.2                           |   0.742   |  0.753         |   20000 tfidf|   rotten  |
AdaBoostClassifier 2000 learning_rate=0.2                           |   0.749   |  0.766         |   20000 tfidf|   rotten  |
AdaBoostClassifier 2000                                             |   0.763   |  0.813         |   20000 tfidf|   rotten  |