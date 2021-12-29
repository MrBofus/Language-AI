################################################################################
#Import libraries
################################################################################

import pandas as pd
from spacy.lang.en import English
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import os

################################################################################
#Defining some variables
################################################################################

TRAIN_ON_RUN = 0 #Training takes a very long time--multiple days of continuous running.
                 #Train the model on a text document called 'text_in' later in the code.

max_length = 1024   #Max length of strings taken in by GPT-2 Model. 
temp = 0.95         #How 'creative' the story is. The lower the number, the more likely
                    #to get stuck in a text loop. The higher the number the more random and
                    #disjointed the story is.
top_k = 50          #Limit the number of Markhov chains to use

################################################################################
#Printing material, including explanation of code and getting starting seeds
#for story to generate around
################################################################################

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('\n\n==================================================================================================================================')
print('==================================================================================================================================\n\n')
print('\t\t\t\tHORROR-BOT: Engagement Mode\n')
print('\t\t\t\ta faster-paced version of regular horror-bot')
print('\n\n==================================================================================================================================')
print('==================================================================================================================================\n\n')
print('\n\n\tLoading...\n\n')

################################################################################
#Definitions of functions called later in code
################################################################################

class storydata(Dataset):  
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.sentences = []

        for row in sentences[0]:
          self.sentences.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))               
        if truncate:
            self.sentences = self.sentences[:20000]
        self.sentences_count = len(self.sentences)
        
    def __len__(self):
        return self.sentences_count

    def __getitem__(self, item):
        return self.sentences[item]

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    device=torch.device("cpu")
    model = model.cpu()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=45, #maximum number of words
    top_p=0.8,
    temperature=temp,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list

def text_generation(test_data):
  generated_tale = []
  for i in range(len(test_data)):
    x = generate(model.to('cpu'), tokenizer, test_data[0][i], entry_count=1)
    generated_tale.append(x)
  return generated_tale

################################################################################
#If training, prepare the text to go into the GPT-2 model. Otherwise, recall the 
#included pre-trained model--trained on the entire novel of Dracula
################################################################################
 
text_in = open("dracula-abridged.txt").read() #Change this to whatever text document to train
                                     #the GPT-2 model on, or leave it as-is to use the
                                     #default.
text_in = text_in.replace('\n',' ')
text_in = text_in.replace('\xa0','')
    
nlp = English()
nlp.add_pipe('sentencizer')
doc = nlp(text_in)
sentences = [sent.text.strip() for sent in doc.sents]
sentences = pd.DataFrame(sentences)
sentences = sentences[sentences[0].apply(lambda x: len(x.split(' ')) < 350)]
    
dataset = storydata(sentences[0], truncate=True, gpt2_type="gpt2") 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if TRAIN_ON_RUN:   
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = train(dataset, model, tokenizer)

else:  
    model = torch.load('model.pt') #Saved model, trained on the entire novel of Dracula
    
################################################################################
#Generate the story using the starting seeds and filter the output to look neat
################################################################################

while 1:
    
    start_str = input('\tInput: ')
    print('\n\n')
    start_str = pd.DataFrame({0:[start_str]})
    generated_story = text_generation(start_str)

    i = 0
    generated_story_str = '\t'

    while i < len(generated_story):
        generated_story_str = generated_story_str + ' ' + str(generated_story[i])
        i += 1

    generated_story = generated_story_str

    generated_story = generated_story.replace('[\'', '')
    generated_story = generated_story.replace('[\"','')
    generated_story = generated_story.replace('\']', '')
    generated_story = generated_story.replace('\"]','\"')
    generated_story = generated_story.replace('\\xa0', '')
    generated_story = generated_story.replace('<|endoftext|>','.')
    generated_story = generated_story.replace('.,','.')
    generated_story = generated_story.replace(',.','.')
    generated_story = generated_story.replace('..','.')
    generated_story = generated_story.replace('      DRACULA     CHAPTER I  JONATHAN HARKERS JOURN','')
    generated_story = generated_story.replace('\\n\\n', ' ')
    generated_story = generated_story.replace('\\n',' ')
    generated_story = generated_story.replace('.\".', '.\"')
    generated_story = generated_story.replace('\\\'','\'')
    generated_story = generated_story.replace('\\\"','\"')

    print('\n\n' + generated_story + '\n\n')