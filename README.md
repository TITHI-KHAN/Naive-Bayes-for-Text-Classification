# Naïve Bayes for Text Classification

# Introduction no Naïve Bayes

❑ Naive Bayes is a classification algorithm based on Bayes' theorem.

❑ It assumes that features are conditionally independent of each other given the class label.

❑ It builds a probability model by estimating the probabilities of features given each class label.

❑ Naive Bayes is primarily used for classification tasks and predicts the class label with the highest probability.

❑ It can handle both categorical and numerical features.

❑ Laplace smoothing is often applied to avoid zero probabilities.

❑ Naive Bayes is computationally efficient and requires a small amount of training data.

❑ It is commonly used in text classification tasks such as spam filtering and sentiment analysis.

❑ There are different variants of Naive Bayes, such as Multinomial Naive Bayes and Gaussian Naive Bayes.

❑ Naive Bayes can still perform well even if the independence assumption is violated to some extent.

Naive Bayes is a Probability-based algorithm. Here, we don't need to do Data Scaling that much. It is used for Classification and mostly used for text data (Email classification, Sentiment analysis, NLP, Spam email classification, etc.). In the case of text, it mainly deals with the frequency (how many times it is there?) of the text.

# Basic Probability

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/39921251-27a0-4432-b3b6-53abaa5f6e2d)

The probability of getting 1 in a dice is 1/6.

The probability of getting 5 in a dice is 1/6.

The probability of getting either head or tail randomly in a coin is 1/2.

The probability/chance of getting WHITE marble randomly is 0/12 = 0 as there is no white marble here.

# Bayesian Classifier 

**Notation:**

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/13fcf964-621e-4d73-8f31-17964706d4bc)

x -> feature (can be a single feature or multiple features)
y -> 0 or 1, positive or negative, multiclass.

Bayesian classifiers have been used in a wide range of applications, including email spam filtering, medical diagnosis, image recognition, and natural language processing.

**Application:**

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/9b7f389f-99b9-407a-bd04-e2887b7a0316)

D1 + D2 -> Total (108)

if there is comma (,), then we will consider the total event (108).

if there is (|), then we will not consider the total event (108). Rather we will consider it individually.

**Bayes rule:**

Bayes' rule, also known as Bayes' theorem or Bayes' law, is a fundamental concept in probability theory. It describes how to update or revise the probability of an event based on new evidence or information. Mathematically, Bayes' rule is represented as:

**P(A|B) = (P(B|A) * P(A)) / P(B)**

In this equation:

▪ P(A|B) is the posterior probability of event A given evidence B. It represents the probability of event A occurring given that evidence B is true.

▪ P(B|A) is the conditional probability of evidence B given event A. It represents the probability of observing evidence B when event A is true.

▪ P(A) is the prior probability of event A. It represents the initial or prior probability of event A before considering any evidence.

▪ P(B) is the probability of evidence B. It represents the overall probability of observing evidence B, regardless of event A.

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/3265a900-2d11-4bf3-b3ac-06d84b5139b0)

*** WE WILL FIND OUT P(y|x): Posterior Probability. This is our Target.

If we use Naive Bayes in the neural network, then that will be called a 'Bayesian Neural Network'.


**Optimality:**

There are **2 ways** to ensure optimality of Naive Bayes.

**1.** Ensuring Good Performance by argmax p(y|x). In the last portion, we considered logarithms for computational power and easy calculation.

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/20a19295-ff79-49d2-aca1-5cc778313af3)

**2.** Reducing Average Loss by argmin AL(x,y).

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/82e29456-1707-49ec-8acb-cb2124e5e172)

# Naïve Bayes: Problems

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/60bb12a9-6529-49a8-b168-d3bb5ea736e4)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/a42e2c59-2b7d-4470-879b-df854bffb16d)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/d54cf118-cfed-49f9-8caf-6a73089389d0)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/e3c48ce2-a78a-4506-9e61-99a8b02adf8c)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/67911423-7fd8-4c45-b455-de0ba10a3cbf)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/7bc98eee-e6a8-4188-b79a-b2a85a086242)

First, we considered P(A) -> Yes for the Bayes Rule P(A|B) = (P(B|A) * P(A)) / P(B) and then we considered P(A) -> No.

Here, P(Yes) = 5/10 = 0.5 and P(No) = 5/10 = 0.5

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/f13774b8-c1a1-4cc7-a3f8-722274e02fb0)

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/e4da0c0c-9558-4b37-b4c2-96f4edaf18a8)


# Stopwords in NLP

Stopwords are commonly used words in a language that are often removed during natural language processing (NLP) tasks because they typically do not carry significant meaning or contribute to the understanding of the text. Removing stopwords helps to reduce noise and focus on more meaningful words.

Here's an overview of stopwords in NLP:

1. **What are stopwords?** Stopwords are words that are considered frequent and common in a given language. They include articles, prepositions, pronouns, conjunctions, and other commonly used words that do not carry specific semantic meaning in isolation.

2. **Purpose of stopwords removal:** Stopwords removal is a common preprocessing step in NLP tasks such as text classification, information retrieval, sentiment analysis, and topic modeling. By eliminating stopwords, the focus is shifted towards words that provide more contextual information and contribute to the underlying meaning of the text.

3. **Examples of stopwords:** In English, common stopwords include "a," "an," "the," "is," "are," "in," "on," "of," "and," "or," "but," "to," "with," and so on. The specific set of stopwords can vary depending on the NLP library or framework being used.

4. **Stopwords in NLP libraries:** Many NLP libraries and frameworks provide built-in sets of stopwords for various languages. For example, NLTK (Natural Language Toolkit) and spaCy, popular Python libraries for NLP, offer predefined sets of stopwords that can be used for stopwords removal.

5. **Custom stopwords:** Depending on the task or domain-specific requirements, you can create your own list of stopwords by including additional words that are frequent in your specific context but do not carry substantial meaning.

6. **Stopwords removal techniques:** In NLP, stopwords can be removed by comparing each word in a text against a set of stopwords and excluding the matching words. This can be done using simple string matching or tokenization techniques, often combined with stemming or lemmatization.

**Remember** that stopwords removal is not always necessary or beneficial for every NLP task. In some cases, stopwords can be informative and should be retained. It's important to consider the specific requirements of your task and the nature of the text data when deciding whether to remove stopwords or not.

*** The more phrased data we will give, the better.

# Gaussian Naive Bayes

Gaussian Naive Bayes assumes that the features follow a Gaussian (normal) distribution. It is suitable for continuous or numerical features. This classifier calculates the mean and standard deviation of each feature for each class and then uses the Gaussian probability density function to estimate the likelihood of a given instance belonging to a specific class.

**Example:** Suppose we have a dataset with two classes, "spam" and "not spam," and two continuous features, "length" and "frequency." Gaussian Naive Bayes would calculate the mean and standard deviation of the "length" and "frequency" features for each class. When a new email arrives, the classifier would estimate the probability of it being spam or not spam based on the Gaussian distribution of the features.

# Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification tasks. It assumes that features follow a multinomial distribution, which is suitable for discrete or count-based features, such as word frequencies or term frequencies. This classifier calculates the probabilities of each feature occurring in each class and uses these probabilities to estimate the likelihood of an instance belonging to a specific class.

**Example:** Consider a sentiment analysis task where we want to classify movie reviews as "positive" or "negative" based on the frequency of words in the review. Multinomial Naive Bayes would calculate the probabilities of each word occurring in positive and negative reviews. Then, given a new review, it would estimate the probability of it being positive or negative based on the frequencies of words in the review.

# Bernoulli Naive Bayes 

Bernoulli Naive Bayes assumes that features are binary or follow a Bernoulli distribution. It is commonly used for binary classification tasks where features are represented as presence or absence indicators. This classifier calculates the probabilities of each feature being present or absent in each class and uses these probabilities to estimate the likelihood of an instance belonging to a specific class.

**Example:** Let's say we want to classify emails as either spam or not spam based on the presence or absence of certain words. Bernoulli Naive Bayes would calculate the probabilities of each word being present or absent in spam and non-spam emails. Then, given a new email, it would estimate the probability of it being spam or not spam based on the presence or absence of those words.

# Confusion Matrix

A confusion matrix is a table that summarizes the performance of a classification model by showing the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). It provides insights into the model's ability to correctly classify instances.

**Example:**

![image](https://github.com/TITHI-KHAN/Naive-Bayes-for-Text-Classification/assets/65033964/0760f51d-26d9-4008-b0f0-4c89298c23ed)

# ROC-AUC Score

ROC (Receiver Operating Characteristic) is a graphical representation of the trade-off between the true positive rate (TPR) and the false positive rate (FPR) at various classification thresholds. The ROC-AUC (Area Under the ROC Curve) score quantifies the overall performance of a classifier in distinguishing between classes. An AUC score of 1 indicates a perfect classifier, while a score of 0.5 suggests random guessing.

**Example:** If the ROC-AUC score is 0.85, it means that the classifier has an 85% chance of ranking a randomly chosen positive instance higher than a randomly chosen negative instance.

These evaluation metrics help assess the performance of the classifiers and provide insights into their effectiveness in classifying instances based on different assumptions and data distributions.


