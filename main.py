import re
import math
import json

class NaiveBayes:
    def __init__(self, options=None):
        self.options = options if options is not None else {}
        self.tokenizer = self.options.get('tokenizer', self.default_tokenizer)
        self.vocabulary = []
        self.total_documents = 0
        self.doc_count = {}
        self.word_count = {}
        self.word_frequency_count = {}
        self.categories = []

    def default_tokenizer(self, text):
        rgx_punctuation = re.compile(r'[^(a-zA-ZA-Яa-я\u4e00-\u9fa50-9_)+\s]')
        text = rgx_punctuation.sub(' ', text)
        text = re.sub(r'[\u4e00-\u9fa5]', lambda x: x.group() + ' ', text)
        return re.findall(r'\w+', text)

    def initialize_category(self, category_name):
        if category_name not in self.categories:
            self.doc_count[category_name] = 0
            self.word_count[category_name] = 0
            self.word_frequency_count[category_name] = {}
            self.categories.append(category_name)

    def learn(self, text, category):
        self.initialize_category(category)
        self.doc_count[category] += 1
        self.total_documents += 1
        tokens = self.tokenizer(text)
        frequency_table = self.frequency_table(tokens)

        for token, frequency_in_text in frequency_table.items():
            if token not in self.vocabulary:
                self.vocabulary.append(token)

            if token not in self.word_frequency_count[category]:
                self.word_frequency_count[category][token] = frequency_in_text
            else:
                self.word_frequency_count[category][token] += frequency_in_text

            self.word_count[category] += frequency_in_text

    def categorize(self, text, probability=False):
        probabilities = self.probabilities(text)
        if probability:
            return probabilities[0][0]
        else:
            return probabilities[0][0]  # Return the category directly, not its attribute

    def probabilities(self, text):
        tokens = self.tokenizer(text)
        frequency_table = self.frequency_table(tokens)
        result = []

        for category in self.categories:
            category_probability = self.doc_count[category] / self.total_documents
            log_probability = math.log(category_probability)

            for token, frequency_in_text in frequency_table.items():
                token_probability = self.token_probability(token, category)
                log_probability += frequency_in_text * math.log(token_probability)

            result.append((category, log_probability))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def token_probability(self, token, category):
        word_frequency_count = self.word_frequency_count[category].get(token, 0)
        word_count = self.word_count[category]
        return (word_frequency_count + 1) / (word_count + len(self.vocabulary))

    def frequency_table(self, tokens):
        frequency_table = {}
        for token in tokens:
            frequency_table[token] = frequency_table.get(token, 0) + 1
        return frequency_table

    def to_json(self, pretty_print=False):
        state = {key: getattr(self, key) for key in STATE_KEYS}
        indent = 2 if pretty_print else None
        return json.dumps(state, indent=indent)

    @staticmethod
    def from_json(json_str):
        json_data = json.loads(json_str)
        options = json_data.get('options', {})
        classifier = NaiveBayes(options=options)
        for key in STATE_KEYS:
            if key not in json_data:
                raise ValueError(f"JSON string is missing an expected property: '{key}'.")
            setattr(classifier, key, json_data[key])
        return classifier

# Define STATE_KEYS
STATE_KEYS = ['categories', 'doc_count', 'total_documents', 'vocabulary', 'word_count', 'word_frequency_count', 'options']

# Keywords for spam detection
spam_keywords = {'money', 'offer', 'free', 'deal', 'discount'}

# Example usage
def main():
    # Initialize classifier
    classifier = NaiveBayes()

    # Train classifier with some examples
    classifier.learn("Get a free laptop today", "spam")
    classifier.learn("Hey, did you see the new discount offer?", "spam")
    classifier.learn("Meeting at 10 am tomorrow", "not spam")
    classifier.learn("Reminder: pay rent by end of the month", "not spam")

    # Classify new emails
    email = input("Enter the email text: ")
    is_spam = classifier.categorize(email) == "spam"

    if is_spam:
        print("This email is likely spam.")
    else:
        print("This email is not spam.")

if __name__ == "__main__":
    main()
