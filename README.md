# Spam Email Detection with MLOps

| | Description |
| ----------- | ----------- |
| Dataset | [Spam email classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) |
| Problem | Spam is a major nuisance for email users. |
| Machine Learning Solution | Machine Learning Natural Language Processing can detect spam in emails with MLOps implementation. |
| Data Processing Method | The data processing method used in this project involves tokenizing the input features (email text). Initially, the text is converted into a sequence of numbers that represent the text, making it easier for the model to understand (Text Vectorization). |
| Model Architecture | MThe model is built using a TextVectorization layer to process input strings into numerical sequences, followed by an Embedding layer that learns the similarity or proximity of words, helping to determine whether a word is negative or positive. There are also 2 hidden layers and 1 output layer. |
| Evaluation Metrics | The metrics used for the model are Binary Accuracy, True Positive, False Positive, True Negative, and False Negative to evaluate the model's performance in classification. |
| Model Performance | The created model performs quite well in making predictions for input news text, and from the training, the model achieves a binary accuracy of over 98%. |
| Deployment Options | The model is deployed on Cloudeka using the Lintasarta Cloudeka's DekaFlexi service. |
| Web App | [spam_ml](https://kianaa19-spam-detection-ml.w3spaces.com)  |
| Monitoring | Monitoring on this system is done using Prometheus and Grafana. Here, only the process of monitoring to display incoming requests to the system is performed, which will display the status of each request made. In this system, there are three statuses displayed: if the request process on the classification system is not found, invalid argument, and the classification process is successful, indicated by "ok". |
