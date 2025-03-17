# README: Enhancing Retrieval Performance through Diversity in Personalized Prompts

## Project Overview
This project explores the impact of diversity on retrieval performance in the context of personalizing input prompts for large language models. As language models are increasingly used in diverse applications, there is a growing need to generate user-specific outputs that consider different backgrounds and preferences. Our approach builds upon prior research on user-profile enrichment, particularly in the LaMP benchmark, by incorporating both similarity augmentation and diversity-enhanced profiles. Our findings demonstrate that integrating a broader range of user information leads to superior retrieval results compared to similarity-based methods alone.

## Installation
To set up the necessary environment, install the required dependencies using the following commands:

```sh
!pip install "peft==0.2.0"
!pip install "transformers==4.27.1" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade --quiet
```

Additionally, install the Hugging Face model:

```python
from transformers import AutoModelForSeq2SeqLM

# Hugging Face Hub model ID
model_id = "google/flan-t5-base"

# Load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
```

## Dataset
We use the LaMP dataset for training and evaluation. The dataset is loaded as follows:

```python
import pandas as pd

data_train = pd.read_json('https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_questions.json')
data_output_train = pd.read_json('https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_outputs.json')
```

## Preprocessing
To preprocess the dataset for fine-tuning, we tokenize the input prompts and expected outputs:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess(input_df):
    model_inputs = tokenizer(input_df['prompt'], max_length=512, truncation=True)
    labels = tokenizer(text_target=input_df['output'], max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
```

## Fine-tuning with LoRA
We fine-tune the model using LoRA (Low-Rank Adaptation) for efficient adaptation:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
```

## Training
We train the model using the following setup:

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    weight_decay=0.001,
    save_total_limit=3,
    num_train_epochs=6,
    predict_with_generate=True,
    push_to_hub=False,
    lr_scheduler_type='linear',
    warmup_steps=100
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data_eval,
)

trainer.train()
```

## Evaluation
After training, we evaluate the model using accuracy and F1 score:

```python
from sklearn.metrics import accuracy_score, f1_score

output_decoded = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
labels = list(final_df_evaluate['output'])

accuracy = accuracy_score(labels, output_decoded)
print("Accuracy of Val Data:", accuracy * 100)
f1 = f1_score(labels, output_decoded, average='weighted')
print("F1 Score of Val Data:", f1)
```

### Results
The model achieves an accuracy of **80.03%** and an F1 score of **0.79**, demonstrating the effectiveness of incorporating diversity-enhanced user profiles.

## Conclusion
This project highlights the importance of incorporating diversity in retrieval-based models. By augmenting user profiles with diverse information, we achieve improved retrieval accuracy and robustness, paving the way for more personalized and adaptable language models.

