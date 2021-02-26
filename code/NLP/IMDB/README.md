## TextAttack Model Card

This `distilbert-base-cased` model was fine-tuned for sequence classification using TextAttack
and the imdb dataset loaded using the `datasets` library. The model was fine-tuned
for 3 epochs with a batch size of 64, a learning
rate of 2e-05, and a maximum sequence length of 128.
Since this was a classification task, the model was trained with a cross-entropy loss function.
The best score the model achieved on this task was 0.79416, as measured by the
eval set accuracy, found after 0 epoch.

For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).
