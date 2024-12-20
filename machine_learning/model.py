import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

# Load Data
user_object = dict()
user_object["fake"] = pd.read_csv("Dataset/fake_twitter_accounts.csv")
user_object["legit"] = pd.read_csv("Dataset/real_twitter_accounts.csv")

# Drop unnecessary columns
columns_to_drop = ["id", "name", "screen_name", "created_at", "lang", "location", 
                   "default_profile", "default_profile_image", "geo_enabled", 
                   "profile_image_url", "profile_banner_url", "profile_use_background_image", 
                   "profile_background_image_url_https", "profile_text_color", 
                   "profile_image_url_https", "profile_sidebar_border_color", 
                   "profile_background_tile", "profile_sidebar_fill_color", 
                   "profile_background_image_url", "profile_background_color", 
                   "profile_link_color", "utc_offset", "protected", "verified", 
                   "dataset", "updated", "description"]

user_object["legit"] = user_object["legit"].drop(columns=columns_to_drop, axis=1)
user_object["fake"] = user_object["fake"].drop(columns=columns_to_drop, axis=1)

# Convert data to numpy arrays
user_object["legit"] = user_object["legit"].values
user_object["fake"] = user_object["fake"].values

# Clean and preprocess data
for key in ["legit", "fake"]:
    for index in range(len(user_object[key])):
        for col in [5, 6]:  # Replace strings in columns 5 and 6 with 1
            if type(user_object[key][index][col]) == str:
                user_object[key][index][col] = 1
    user_object[key] = user_object[key].astype(np.float64)
    user_object[key][np.isnan(user_object[key])] = 0  # Replace NaNs with 0

# Prepare input (X) and output (Y)
X = np.zeros((len(user_object["fake"]) + len(user_object["legit"]), 7))
Y = np.zeros(len(user_object["fake"]) + len(user_object["legit"]))

for index in range(len(user_object["legit"])):
    X[index] = user_object["legit"][index] / max(user_object["legit"][index])
    Y[index] = -1  # Legit accounts

for index in range(len(user_object["fake"])):
    bound = max(user_object["fake"][index]) if max(user_object["fake"][index]) != 0 else 1
    X[len(user_object["legit"]) + index] = user_object["fake"][index] / bound
    Y[len(user_object["legit"]) + index] = 1  # Fake accounts

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.24, random_state=42)

# Train an SVM model
print("Training SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF kernel
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Reds):
    target_names = ['Fake', 'Real']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm)
plt.show()
