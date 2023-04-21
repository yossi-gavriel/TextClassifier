from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from TextClassifier import TextClassifier
#from datasets import load_dataset
from datasets import load_dataset

# Load the IMDB movie reviews dataset from Hugging Face Datasets
dataset = datasets.load_dataset("imdb")

# Split into training and testing data
train_data, test_data = train_test_split(dataset["train"], test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

# Fit the pipeline to the training data
pipeline.fit(train_data['text'], train_data['label'])

# Evaluate the pipeline on the testing data
accuracy = pipeline.score(test_data['text'], test_data['label'])
print(f"Accuracy: {accuracy}")

# Train the text classifier separately
model_name = 'distilbert-base-uncased'
num_labels = 2
text_classifier = TextClassifier(model_name, num_labels)
text_classifier.train(train_data['text'], train_data['label'])

# Evaluate the text classifier on the testing data
predictions = text_classifier.predict(test_data['text'])
text_classifier_accuracy = sum([1 if pred.argmax() == label else 0 for pred, label in zip(predictions, test_data['label'])]) / len(predictions)
print(f"Text Classifier Accuracy: {text_classifier_accuracy}")