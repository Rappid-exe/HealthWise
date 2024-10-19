import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import openai

# Your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Example transaction data (dummy data for illustration)
data = {
    'category': ['gym', 'fast_food', 'gym', 'fast_food', 'healthy_groceries', 'alcohol'],
    'amount': [50, 20, 70, 15, 100, 30],
    'premium_adjustment': [-5, 10, -7, 8, -10, 12]  # Premium adjustment in percentage
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Map categories to numerical values
category_mapping = {
    'gym': 0,
    'healthy_groceries': 1,
    'fast_food': 2,
    'alcohol': 3
}

df['category_code'] = df['category'].map(category_mapping)

# Normalize the 'amount' feature
scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])

# Features and target variable
X = df[['category_code', 'amount_scaled']]  # Category and scaled amount are the features
y = df['premium_adjustment']  # The target variable is the premium adjustment

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# GPT-based suggestion function
def get_gpt_suggestions(message):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Provide feedback based on accuracy
if accuracy < 0.5:
    feedback_prompt = "My decision tree model performed poorly with an accuracy of {:.2f}. Can you suggest ways to improve it?".format(accuracy * 100)
    suggestion = get_gpt_suggestions(feedback_prompt)
    print("Suggestion from GPT to improve the model:\n", suggestion)
else:
    appreciation_prompt = "My decision tree model performed well with an accuracy of {:.2f}. Please provide some positive feedback.".format(accuracy * 100)
    appreciation = get_gpt_suggestions(appreciation_prompt)
    print("Appreciation from GPT:\n", appreciation)

# Example function to predict premium adjustment for a new transaction
def predict_premium_adjustment(category, amount):
    category_code = category_mapping.get(category, -1)
    if category_code == -1:
        return "Unknown category"
    
    # Scale the amount to match the trained model's input format
    amount_scaled = scaler.transform([[amount]])[0][0]
    
    # Create a DataFrame with proper feature names for prediction
    input_data = pd.DataFrame([[category_code, amount_scaled]], columns=['category_code', 'amount_scaled'])
    return clf.predict(input_data)[0]

# Example usage
new_transaction = {
    'category': 'gym',
    'amount': 60
}

predicted_adjustment = predict_premium_adjustment(new_transaction['category'], new_transaction['amount'])
print(f"Predicted premium adjustment: {predicted_adjustment}%")

# Feedback on premium adjustment
if predicted_adjustment > 0:
    improvement_prompt = "The premium adjustment is {:.2f}%. What can I do to improve my financial health?".format(predicted_adjustment)
    suggestion = get_gpt_suggestions(improvement_prompt)
    print("Suggestion from GPT:\n", suggestion)
else:
    positive_feedback_prompt = "The premium adjustment is {:.2f}%. Please provide positive feedback on my spending behavior.".format(predicted_adjustment)
    appreciation = get_gpt_suggestions(positive_feedback_prompt)
    print("Appreciation from GPT:\n", appreciation)
