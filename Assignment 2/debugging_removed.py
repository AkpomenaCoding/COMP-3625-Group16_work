"""
This file can be IGNORED, it was used as a dump file for debugging done while writing and testing other code
"""

"""
print(my_x.shape)

print(my_y.shape)
print(my_y.mean())
"""


"""
print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.mean(), y_val.mean(), y_test.mean())
"""

"""
model.fit(x_train, y_train)
p_val = model.predict_proba(x_val)[:,1]

print(p_val[:5])
print(p_val.min(), p_val.max(), p_val.mean())
"""
"""
# log loss compares our how good our predicitons were, while brier the average of our errors in predicting it.,
# the baseline is 0.335 which is the general average from my tests before. 
# y_baseline is just array with the same length as y_val filled with the mean 0.335,
# after log_loss comapres it seems that we're not to far off our baseline which I see as good. 
print(log_loss(y_val, p_val), log_loss(y_val, y_baseline))  
print(brier_score_loss(y_val, p_val), brier_score_loss(y_val, y_baseline))
"""
"""
candidates = {
    "logistic": LogisticRegression(max_iter=1000),
    "boosting": HistGradientBoostingClassifier(random_state=0),
    "boosting3": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0),
    "boosting8": HistGradientBoostingClassifier(max_depth=8),
    "boosting10": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.1, random_state=0)
}
"""

"""
for name, model in candidates.items():
    model.fit(x_train, y_train)
    p_val = model.predict_proba(x_val)[:,1]
    curr_log_loss = log_loss(y_val, p_val)

    if (curr_log_loss < best_score):
        best_score = curr_log_loss
        best_name = name
    # print(name, curr_log_loss)

"""

"""
print("Best: ", best_name, best_score)
"""

"""
print("----- Against P_test ------")
print(log_loss(y_test, p_test))
print(brier_score_loss(y_test, p_test))

# this baseline is for the test set. 
new_y_baseline = np.full(len(y_test), y_train.mean())
print("----- Against Baseline ------")
print(log_loss(y_test, new_y_baseline))
print(brier_score_loss(y_test, new_y_baseline))
"""

"""
print("Coverage shows, ", total)
print("Expected profit, ", profit[desired_customer].sum())
"""